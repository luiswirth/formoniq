//! Geometry: the metric a mesh carries, in any of its equivalent forms.
//!
//! The geometry of a simplicial manifold is intrinsic -- it is fully carried by
//! the pseudo-Riemannian metric of each cell, of any signature, with no
//! reference to coordinates. The
//! [`Geometry`] trait captures exactly that: `cell_metric` gives a cell its flat
//! metric tensor.
//! It is implemented by every geometry representation, related by the
//! derivation chain coords $->$ squared edge lengths $->$ per-cell metric:
//!
//! - [`MeshLengthsSq`]: intrinsic Regge geometry, signed squared edge lengths
//!   (grade-1 data), of any signature.
//! - [`CellGramians`]: the metric tensors themselves as per-cell data (grade
//!   n) -- the most local, coordinate-free geometry, living natively on the
//!   cell skeleton with no need of a global edge indexing.
//! - [`MeshCoords`](crate::geometry::coord::mesh::MeshCoords): an extrinsic
//!   embedding (grade-0 data), which *induces* a metric. It implements
//!   [`Geometry`] from the [`coord`](crate::geometry::coord) module, one layer
//!   up: an embedding knows about the metric it induces, but the metric layer
//!   knows nothing of embeddings, and must not.

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

/// The intrinsic geometry of a mesh: the pseudo-Riemannian metric of each
/// cell, which is all the metric information the manifold carries. The
/// signature is the metric's own: a Riemannian mesh and a Lorentzian
/// spacetime mesh implement the same trait.
pub trait Geometry {
  /// The flat metric tensor of a cell. The [`Cell`] witness is the
  /// precondition: only a top-dimensional simplex has a metric here.
  fn cell_metric(&self, cell: Cell) -> Metric;
}

impl Geometry for MeshLengthsSq {
  fn cell_metric(&self, cell: Cell) -> Metric {
    self.simplex_lengths_sq(cell.get()).metric()
  }
}

/// The per-cell metric tensors as grade-n data on the mesh: the most local,
/// coordinate-free geometry. Each cell independently carries its flat metric,
/// so this is defined on the cell skeleton alone.
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

  /// Sample the per-cell metrics of any other geometry over a complex.
  pub fn from_geometry(topology: &Complex, geometry: &impl Geometry) -> Self {
    let metrics = topology
      .cells()
      .handle_iter()
      .map(|cell| geometry.cell_metric(cell))
      .collect();
    Self::new(topology.dim(), metrics)
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

impl Geometry for CellGramians {
  fn cell_metric(&self, cell: Cell) -> Metric {
    self.metrics[cell.get()].clone()
  }
}

/// Regge signed squared edge lengths from a cell's metric tensor, on any
/// signature.
pub fn simplex_lengths_sq_of(metric: &Metric) -> SimplexLengthsSq {
  SimplexLengthsSq::from_metric_tensor(metric.vector_gramian())
}
