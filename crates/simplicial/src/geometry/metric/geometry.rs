//! The intrinsic geometry of a mesh, in its two coordinate-free forms.
//!
//! The geometry of a simplicial manifold is intrinsic: it is fully carried by
//! the pseudo-Riemannian metric of each simplex, of any signature, with no
//! reference to coordinates. Two representations carry it, related by the
//! derivation chain squared edge lengths $->$ per-simplex metric:
//!
//! - [`MeshLengthsSq`]: the Regge primitive, signed squared edge lengths
//!   (grade-1 data) on the 1-skeleton, of any signature. It is *total over
//!   every grade*: the metric of any subsimplex is the Gramian of that
//!   simplex's own edges ([`MeshLengthsSq::simplex_metric`]), so this is the
//!   representation the whole engine speaks, and the one boundary traces and
//!   curvature read.
//! - [`CellGramians`]: the per-cell metric tensors as grade-n data -- the
//!   materialized cell column of the derivation chain, living natively on the
//!   cell skeleton with no need of a global edge indexing. A convenience for a
//!   source that arrives as raw per-cell metrics and the intermediary of
//!   [`refine_gramians`](crate::topology::refine::Subdivision::refine_gramians);
//!   it converts back to edge lengths losslessly on a face-consistent geometry.
//!
//! An embedding ([`MeshCoords`](crate::geometry::coord::mesh::MeshCoords),
//! grade-0 data) is a *third* source, but it lives one layer up in
//! [`coord`](crate::geometry::coord): an embedding induces a metric, the metric
//! layer knows nothing of embeddings and must not.
//!
//! There is no trait unifying the representations. Each answers "the metric of
//! a simplex" concretely, and a source that arrives in another form converts to
//! edge lengths (the primitive) or per-cell metrics at the boundary of the API,
//! not through runtime dispatch on the hot path.

use super::{mesh::MeshLengthsSq, simplex::SimplexLengthsSq};
use crate::{
  topology::{
    complex::Complex,
    data::{SkeletonData, SkeletonVec},
    role::Cell,
  },
  Dim,
};

use gramian::Metric;

#[cfg(feature = "serde")]
use std::{io, path::Path};

/// The per-cell metric tensors as grade-n data on the mesh: the most local,
/// coordinate-free geometry. Each cell independently carries its flat metric,
/// so this is defined on the cell skeleton alone, with no global edge indexing.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CellGramians {
  metrics: SkeletonVec<Metric>,
}
impl CellGramians {
  pub fn new(dim: Dim, metrics: Vec<Metric>) -> Self {
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
  /// any signature. Each cell's metric gives the signed squared lengths of
  /// its own edges ([`simplex_lengths_sq_of`]), scattered to the global
  /// edges; an edge shared between cells receives the same value from each
  /// (the metrics agree on a shared face), so the result is well defined. A
  /// 0-manifold has an empty edge skeleton and yields the empty vector.
  pub fn to_edge_lengths_sq(&self, topology: &Complex) -> MeshLengthsSq {
    if topology.dim() == 0 {
      return MeshLengthsSq::new_unchecked(crate::linalg::Vector::zeros(0));
    }
    let mut edge_lengths_sq = crate::linalg::Vector::zeros(topology.edges().len());
    for cell in topology.cells().handle_iter() {
      let lengths_sq = simplex_lengths_sq_of(&self.metrics[cell.get()]);
      for (local, edge) in cell.get().edges().enumerate() {
        edge_lengths_sq[edge.kidx()] = lengths_sq.length_sq(local);
      }
    }
    MeshLengthsSq::new_unchecked(edge_lengths_sq)
  }

  /// Whether this could be the per-cell geometry of `topology`: one metric per
  /// simplex of the grade it was built for.
  pub fn is_compatible_with(&self, topology: &Complex) -> bool {
    self.metrics.len() == topology.skeleton(self.metrics.grade()).len()
  }

  #[cfg(feature = "serde")]
  pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
    crate::io::cbor::save_cbor(self, path)
  }
  #[cfg(feature = "serde")]
  pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
    crate::io::cbor::load_cbor(path)
  }
}

/// Regge signed squared edge lengths from a cell's metric tensor, on any
/// signature.
pub fn simplex_lengths_sq_of(metric: &Metric) -> SimplexLengthsSq {
  SimplexLengthsSq::from_metric_tensor(metric.vector_gramian())
}
