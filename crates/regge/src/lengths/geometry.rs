//! The intrinsic geometry of a mesh, in its two coordinate-free forms.
//!
//! The geometry of a simplicial manifold is intrinsic: it is fully carried by
//! the pseudo-Riemannian metric of each simplex, of any signature, with no
//! reference to coordinates. Two representations carry it, related by the
//! derivation chain squared edge lengths $->$ per-simplex metric:
//!
//! - [`MeshLengthsSq`]: the Regge primitive, signed squared edge lengths
//!   (grade-1 data) on the 1-skeleton, of any signature. It is total over
//!   every grade: the metric of any subsimplex is the Gramian of that
//!   simplex's own edges ([`MeshLengthsSq::simplex_metric`]), so this is the
//!   representation the whole engine speaks, and the one boundary traces and
//!   curvature read.
//! - [`CellGramians`]: the per-cell metric tensors as grade-n data, the
//!   materialized cell column of the derivation chain, living natively on the
//!   cell skeleton with no need of a global edge indexing. A convenience for a
//!   source that arrives as raw per-cell metrics and the intermediary of
//!   [`refine_gramians`](crate::refine::SubdivisionExt::refine_gramians);
//!   it converts back to edge lengths losslessly on a Regge-conforming
//!   geometry ([`CellGramians::is_regge_conforming`]), and only there. Not to be
//!   confused with the face consistency of a
//!   [`CellOrdering`](simplicial::topology::ordering::CellOrdering), which is about
//!   how shared faces are subdivided, not about what metric they carry.
//!
//! An embedding ([`MeshCoords`](crate::coord::mesh::MeshCoords),
//! grade-0 data) is a third source, but it lives one layer up in
//! [`coord`](crate::coord): an embedding induces a metric, the metric
//! layer knows nothing of embeddings and must not.
//!
//! There is no trait unifying the representations. Each answers "the metric of
//! a simplex" concretely, and a source that arrives in another form converts to
//! edge lengths (the primitive) or per-cell metrics at the boundary of the API,
//! not through runtime dispatch on the hot path.

use super::{LengthsSq, mesh::MeshLengthsSq, simplex::SimplexLengthsSq};
use simplicial::{
  Dim,
  topology::{
    complex::Complex,
    data::{SkeletonData, SkeletonVec},
    role::Cell,
  },
};

use metric::Metric;

#[cfg(feature = "serde")]
use std::{io, path::Path};

/// Relative tolerance on the tangential-tangential agreement of two cells at a
/// shared edge, taken against the scale of the cell being read. The round trip
/// through a metric tensor is polarization, which is exact in real arithmetic
/// and rounds in floating point, so conforming data agrees to a few ulp rather
/// than bitwise.
const REGGE_CONFORMITY_TOL: f64 = 1e-9;

/// The per-cell metric tensors as grade-n data on the mesh: the most local,
/// coordinate-free geometry. Each cell independently carries its flat metric,
/// so this is defined on the cell skeleton alone, with no global edge indexing.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CellGramians {
  metrics: SkeletonVec<Metric>,
}
impl CellGramians {
  pub fn new(dim: impl Into<Dim>, metrics: Vec<Metric>) -> Self {
    Self {
      metrics: SkeletonVec::new(dim, metrics),
    }
  }

  /// Materialize the per-cell metrics of the Regge (edge-length) geometry: the
  /// $"lengths" -> "metric"$ leg of the derivation chain, read at mesh scope.
  pub fn from_lengths(topology: &Complex, lengths: &MeshLengthsSq) -> Self {
    let metrics = topology
      .cells()
      .handle_iter()
      .map(|cell| lengths.cell_metric(cell))
      .collect();
    Self::new(topology.dim(), metrics)
  }

  /// The flat metric tensor of a cell: a direct lookup, no derivation.
  pub fn cell_metric(&self, cell: Cell) -> Metric {
    self.metrics[cell.get()].clone()
  }

  pub fn metrics(&self) -> &SkeletonVec<Metric> {
    &self.metrics
  }

  /// The Regge squared edge lengths this geometry induces: the missing
  /// metric $->$ lengths leg of the derivation chain, read at mesh scope, on
  /// any signature. A 0-manifold has an empty edge skeleton and yields the
  /// empty vector.
  ///
  /// Defined only on Regge-conforming data ([`Self::is_regge_conforming`]),
  /// and panics otherwise. Per-cell metrics are the broken symmetric-tensor
  /// field, strictly larger than the Regge space: off the conforming subspace
  /// a shared edge has two lengths and no canonical choice between them, so
  /// there is nothing to return. Use [`Self::try_to_edge_lengths_sq`] where
  /// the conformity is not already known.
  pub fn to_edge_lengths_sq(&self, topology: &Complex) -> MeshLengthsSq {
    self
      .try_to_edge_lengths_sq(topology)
      .expect("Per-cell metrics must be Regge-conforming to induce edge lengths.")
  }

  /// [`Self::to_edge_lengths_sq`] where the conformity is a question rather
  /// than a precondition: `None` on data that disagrees on a shared edge,
  /// never a length that silently favors one cell over another.
  pub fn try_to_edge_lengths_sq(&self, topology: &Complex) -> Option<MeshLengthsSq> {
    let nedges = topology.skeleton(Dim::ONE).len();
    let mut edge_lengths_sq = simplicial::linalg::Vector::zeros(nedges);
    let mut written = vec![false; nedges];
    for cell in topology.cells().handle_iter() {
      let lengths_sq = SimplexLengthsSq::from_metric(&self.metrics[cell.get()]);
      let scale = (0..lengths_sq.nedges())
        .map(|local| lengths_sq.length_sq(local).abs())
        .fold(0.0, f64::max);
      for (local, edge) in cell.get().edges().enumerate() {
        let length_sq = lengths_sq.length_sq(local);
        let iedge = edge.kidx();
        if written[iedge] {
          let written_length_sq: f64 = edge_lengths_sq[iedge];
          if (written_length_sq - length_sq).abs() > REGGE_CONFORMITY_TOL * scale {
            return None;
          }
        } else {
          edge_lengths_sq[iedge] = length_sq;
          written[iedge] = true;
        }
      }
    }
    // Every edge length was read off a cell's own lengths, whose
    // non-degeneracy is that type's constructor contract, and the loop above
    // established that two cells agree wherever they share an edge.
    Some(MeshLengthsSq::new(edge_lengths_sq))
  }

  /// Whether these per-cell metrics lie in the Regge space: whether two cells
  /// sharing an edge induce the same signed squared length on it.
  ///
  /// This is tangential-tangential continuity, the conformity condition of
  /// the Regge element, in the form the edge degrees of freedom take. By
  /// polarization, agreeing on the squared lengths of a shared face's edges is
  /// the same as agreeing on $g(u, v)$ for all $u, v$ tangent to that face,
  /// while the normal components stay free to jump. It is what makes
  /// [`MeshLengthsSq`] able to carry the geometry at all: a single value per
  /// edge of the global 1-skeleton is a conforming symmetric-tensor field,
  /// with conformity structural rather than enforced.
  ///
  /// Compared at a tolerance relative to each cell's own scale, so a null edge
  /// of an indefinite metric is judged against the size of its cell rather
  /// than against zero.
  pub fn is_regge_conforming(&self, topology: &Complex) -> bool {
    self.try_to_edge_lengths_sq(topology).is_some()
  }

  /// Whether this could be the per-cell geometry of `topology`: one metric per
  /// simplex of the grade it was built for.
  pub fn is_compatible_with(&self, topology: &Complex) -> bool {
    self.metrics.len() == topology.skeleton(self.metrics.grade()).len()
  }

  #[cfg(feature = "serde")]
  pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
    simplicial::io::cbor::save_cbor(self, path)
  }
  #[cfg(feature = "serde")]
  pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
    simplicial::io::cbor::load_cbor(path)
  }
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::mesher::cartesian::CartesianGrid;
  use metric::Metric;
  use multialgebra::Variance;

  /// Tangential-tangential continuity is what makes the metric $->$ lengths
  /// leg well defined at all, so a geometry that violates it has to be refused
  /// rather than resolved in favor of whichever cell happens to be visited
  /// last.
  #[test]
  fn non_conforming_cell_metrics_induce_no_edge_lengths() {
    for dim in 2..=3 {
      let dim = Dim::from(dim);
      let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
      let lengths = coords.to_edge_lengths_sq(&topology);

      let conforming = CellGramians::from_lengths(&topology, &lengths);
      assert!(conforming.is_regge_conforming(&topology));
      approx::assert_relative_eq!(
        conforming.to_edge_lengths_sq(&topology).vector(),
        lengths.vector(),
        epsilon = 1e-12
      );

      // Stretching one cell alone leaves its neighbors' lengths on the faces
      // they share untouched, which is exactly a tangential-tangential jump.
      let mut metrics: Vec<Metric> = topology
        .cells()
        .handle_iter()
        .map(|cell| lengths.cell_metric(cell))
        .collect();
      metrics[0] = Metric::new(Variance::Covariant, metrics[0].matrix() * 1.5);
      let broken = CellGramians::new(topology.dim(), metrics);

      assert!(!broken.is_regge_conforming(&topology));
      assert!(broken.try_to_edge_lengths_sq(&topology).is_none());
    }
  }
}
