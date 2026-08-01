//! The geometry a subcomplex inherits from its parent.
//!
//! `simplicial`'s `Subcomplex` is pure combinatorics: which parent
//! simplices it consists of. Restricting the parent's geometry to
//! it needs a metric, so it reaches down from here, as an extension trait
//! because the trace reads as an operation on the subcomplex.

use simplicial::topology::subcomplex::Subcomplex;

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
  fn trace_lengths_sq(&self, parent: &MeshLengthsSq) -> MeshLengthsSq {
    // A 0-dimensional boundary (of a 1d mesh) carries no edges, and its
    // inclusion at grade 1 is the empty selection, not an index past the end.
    let lengths_sq: Vec<f64> = self
      .parent_kidxs(1)
      .iter()
      .map(|&iedge| parent[iedge])
      .collect();
    // A cell of the subcomplex is a face of a parent cell, and a face of a
    // non-degenerate simplex is non-degenerate.
    MeshLengthsSq::new(lengths_sq.into())
  }

  fn trace_coords(&self, parent: &MeshCoords) -> MeshCoords {
    parent.select(self.parent_kidxs(0))
  }
}
