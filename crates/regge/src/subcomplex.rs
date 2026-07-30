//! The geometry a subcomplex inherits from its parent.
//!
//! `simplicial`'s `Subcomplex` is pure combinatorics: which parent
//! simplices it consists of. Restricting the parent's geometry to
//! it needs a metric, so it reaches down from here, as an extension trait
//! because the trace reads as an operation on the subcomplex.

use multiindex::Dim;
use simplicial::{linalg::Matrix, topology::subcomplex::Subcomplex};

use crate::{coord::mesh::MeshCoords, lengths::mesh::MeshLengthsSq};

/// Restricting a parent's geometry to a subcomplex.
pub trait SubcomplexExt {
  /// The induced geometry: parent squared edge lengths restricted to the
  /// boundary. A pure data restriction, total on any signature; on an
  /// indefinite parent a null facet carries degenerate data, which surfaces
  /// where a facet metric is actually built from it, not here.
  fn trace_lengths_sq(&self, parent: &MeshLengthsSq) -> MeshLengthsSq;

  /// The vertex coordinates restricted to the boundary.
  fn trace_coords(&self, parent: &MeshCoords) -> MeshCoords;
}

impl SubcomplexExt for Subcomplex {
  /// The induced geometry: parent squared edge lengths restricted to the
  /// boundary. A pure data restriction, total on any signature; on an
  /// indefinite parent a null facet carries degenerate data, which surfaces
  /// where a facet metric is actually built from it, not here.
  fn trace_lengths_sq(&self, parent: &MeshLengthsSq) -> MeshLengthsSq {
    // A 0-dimensional boundary (of a 1d mesh) has no edges.
    let lengths_sq: Vec<f64> = if self.dim() == 0 {
      Vec::new()
    } else {
      self
        .parent_kidxs(Dim::ONE)
        .iter()
        .map(|&iedge| parent[iedge])
        .collect()
    };
    MeshLengthsSq::new_unchecked(lengths_sq.into())
  }

  /// The vertex coordinates restricted to the boundary.
  fn trace_coords(&self, parent: &MeshCoords) -> MeshCoords {
    let columns: Vec<_> = self
      .parent_kidxs(Dim::ZERO)
      .iter()
      .map(|&ivertex| parent.matrix().column(ivertex))
      .collect();
    MeshCoords::with_ambient(Matrix::from_columns(&columns), parent.ambient().clone())
  }
}
