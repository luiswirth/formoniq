//! Transporting geometry across a [`Subdivision`].
//!
//! A [`Subdivision`] carries only topology and affine provenance; the geometry
//! is followed here, on the geometry side of the topology/geometry split. Every
//! child cell is an affine subcell of a flat parent, so its geometry is an exact
//! pullback of the parent's, so no approximation is introduced by refining, on
//! either representation:
//!
//! - Intrinsic ([`Subdivision::refine_gramians`]): each child's metric is
//!   the parent metric pulled back along the child's Jacobian. This is the
//!   coordinate-free refinement, and the primitive the extrinsic case must
//!   agree with. It is the functor and not a formula resembling it:
//!   [`Metric::pullback`](metric::Metric::pullback) is `Tensor::pullback` on the metric's $"Sym"^2$
//!   reading, which `metric`'s laws state, so refining a geometry is the same
//!   operation as transporting any other covariant tensor.
//! - Extrinsic ([`MeshCoords::refine`]): each new vertex is placed by the
//!   affine combination of coarse vertices recorded in its
//!   [`VertexBirth`](simplicial::topology::refine::VertexBirth). An embedding is not
//!   needed to refine: it is refined only because visualization and I/O want
//!   one, and the metric it induces equals the intrinsic refinement, which is
//!   the law that ties the two.

use crate::{
  coord::{Coord, mesh::MeshCoords},
  lengths::{CellGramians, mesh::MeshLengthsSq},
};
use simplicial::topology::{complex::Complex, refine::Subdivision};

/// The geometric half of a refinement.
///
/// `simplicial`'s [`Subdivision`] is the combinatorial record of which child
/// belongs to which parent; pulling a geometry back through it needs a metric,
/// so it reaches down from here as an extension.
pub trait SubdivisionExt {
  /// Refine per-cell metrics: each child carries the pullback of its parent
  /// cell's metric along the child's affine Jacobian. Exact and coordinate-free
  ///, the intrinsic refinement of any geometry, once reduced to its per-cell
  /// metrics ([`CellGramians`]).
  fn refine_gramians(&self, coarse: &CellGramians) -> CellGramians;
}

impl SubdivisionExt for Subdivision {
  fn refine_gramians(&self, coarse: &CellGramians) -> CellGramians {
    let metrics = self
      .children()
      .values()
      .iter()
      .map(|child| coarse.metrics()[child.parent].pullback(&child.jacobian))
      .collect();
    CellGramians::new(self.complex().dim(), metrics)
  }
}

impl MeshLengthsSq {
  /// Refine intrinsic Regge geometry: the refined squared edge lengths of the flat
  /// subdivision. Routed through the metric primitive
  /// ([`Subdivision::refine_gramians`]) rather than reimplemented, coarse
  /// lengths give per-cell metrics, those are pulled back onto the children, and
  /// the fine metrics are read back as edge lengths. Exact; refinement of a flat
  /// cell introduces no geometric error.
  pub fn refine(&self, sub: &Subdivision, coarse: &Complex) -> MeshLengthsSq {
    let coarse_g = CellGramians::from_lengths(coarse, self);
    sub
      .refine_gramians(&coarse_g)
      .to_edge_lengths_sq(sub.complex())
  }
}

impl MeshCoords {
  /// Refine an embedding: the coarse vertices keep their coordinates and label,
  /// and each new vertex is the affine combination of coarse vertices its
  /// [`VertexBirth`](simplicial::topology::refine::VertexBirth) records. Extrinsic,
  /// for I/O and visualization. The intrinsic refinement is
  /// [`Subdivision::refine_gramians`].
  pub fn refine(&self, sub: &Subdivision) -> MeshCoords {
    assert_eq!(
      self.nvertices(),
      sub.ncoarse_vertices(),
      "coordinates must match the coarse mesh being refined"
    );
    let ambient = self.dim();
    let mut matrix = simplicial::linalg::Matrix::zeros(ambient.index(), sub.nvertices());
    matrix
      .view_range_mut(.., 0..sub.ncoarse_vertices())
      .copy_from(self.matrix());
    for (i, birth) in sub.new_births().iter().enumerate() {
      // The weights of a birth sum to one, so the new vertex is where it is
      // independently of any origin: the one combination of points there is.
      let born = Coord::affine_combination(
        birth
          .combination
          .iter()
          .map(|&(vertex, weight)| (weight, self.coord(vertex))),
      );
      matrix.set_column(sub.ncoarse_vertices() + i, born.vector());
    }
    MeshCoords::with_ambient(matrix, self.ambient().clone())
  }
}
