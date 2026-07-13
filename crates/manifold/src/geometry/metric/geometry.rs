//! Geometry: the metric a mesh carries, in any of its equivalent forms.
//!
//! Finite element exterior calculus is intrinsic -- assembly consumes only the
//! Riemannian metric of each cell, never coordinates. The [`Geometry`] trait
//! captures exactly that: `cell_metric` gives a cell its flat metric tensor.
//! It is implemented by every geometry representation, related by the
//! derivation chain coords $->$ edge lengths $->$ per-cell metric:
//!
//! - [`MeshCoords`](crate::geometry::coord::mesh::MeshCoords): extrinsic
//!   embedding (grade-0 data), which induces the metric.
//! - [`MeshLengths`](super::mesh::MeshLengths): intrinsic Regge edge lengths
//!   (grade-1 data).
//! - [`CellGramians`]: the metric tensors themselves as per-cell data (grade
//!   n) -- the most local, coordinate-free geometry, living natively on the
//!   cell skeleton with no need of a global edge indexing.

use super::{mesh::MeshLengths, simplex::SimplexLengths};
use crate::{
  geometry::coord::{mesh::MeshCoords, simplex::SimplexRefExt},
  topology::{complex::Complex, data::SkeletonVec, handle::SimplexRef},
  Dim,
};

use common::gramian::RiemannianMetric;

/// The intrinsic geometry of a mesh: the Riemannian metric of each cell, which
/// is all that FEEC assembly needs.
pub trait Geometry {
  /// The flat metric tensor of a cell.
  fn cell_metric(&self, cell: SimplexRef) -> RiemannianMetric;
}

impl Geometry for MeshLengths {
  fn cell_metric(&self, cell: SimplexRef) -> RiemannianMetric {
    self.simplex_lengths(cell).riemannian_metric()
  }
}

impl Geometry for MeshCoords {
  fn cell_metric(&self, cell: SimplexRef) -> RiemannianMetric {
    RiemannianMetric::new(cell.coord_simplex(self).metric_tensor())
  }
}

/// The per-cell metric tensors as grade-n data on the mesh: the most local,
/// coordinate-free geometry. Each cell independently carries its flat metric,
/// so this is defined on the cell skeleton alone.
#[derive(Debug, Clone)]
pub struct CellGramians {
  metrics: SkeletonVec<RiemannianMetric>,
}
impl CellGramians {
  pub fn new(dim: Dim, metrics: Vec<RiemannianMetric>) -> Self {
    Self {
      metrics: SkeletonVec::new(dim, metrics),
    }
  }

  /// Sample the per-cell metrics of any other geometry over a complex.
  pub fn from_geometry(topology: &Complex, geometry: &impl Geometry) -> Self {
    let metrics = topology
      .cells()
      .handle_iter()
      .map(|cell| geometry.cell_metric(cell))
      .collect();
    Self::new(topology.dim(), metrics)
  }

  pub fn metrics(&self) -> &SkeletonVec<RiemannianMetric> {
    &self.metrics
  }
}

impl Geometry for CellGramians {
  fn cell_metric(&self, cell: SimplexRef) -> RiemannianMetric {
    self.metrics[cell].clone()
  }
}

/// Regge edge lengths from a cell's metric tensor.
pub fn simplex_lengths_of(metric: &RiemannianMetric) -> SimplexLengths {
  SimplexLengths::from_metric_tensor(metric.vector_gramian())
}
