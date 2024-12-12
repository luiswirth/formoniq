//! Module for creating simplicial manifolds
//!
//! This module is concerned with the "raw" data required to fully specify and
//! define a coordinate-free simplicial Riemannian manifold.
//! This includes both topological and geometric data.
//! The structs in this module try to be minimal in the sense that they don't
//! include any information that is redundant or can be derived from other fields.

use super::{complex::Complex, Manifold};
use crate::{
  mesh::{
    geometry::EdgeLengths,
    simplicial::{OrientedVertplex, SimplexExt as _, SortedVertplex},
  },
  Dim,
};

use std::collections::HashMap;

/// The data defining a simplicial Riemanninan manifold.
pub struct RawSimplicialManifold {
  /// number of vertices
  nvertices: usize,
  /// A mapping [`CellIdx`] -> [`RawSimplexTopology`].
  /// Defines topology (connectivity + orientation) and global numbering/order of cells.
  facets: Vec<OrientedVertplex>,
  /// A mapping [`SortedSimplex`] -> [`Length`].
  /// Defines geometry of the manifold through the lengths of all edges.
  edge_lengths: HashMap<SortedVertplex, f64>,
}
impl RawSimplicialManifold {
  pub fn new(
    nvertices: usize,
    cells: Vec<OrientedVertplex>,
    edge_lengths: HashMap<SortedVertplex, f64>,
  ) -> Self {
    Self {
      nvertices,
      facets: cells,
      edge_lengths,
    }
  }
  pub fn dim(&self) -> Dim {
    self.facets[0].dim()
  }
  pub fn nvertices(&self) -> usize {
    self.nvertices
  }
}

impl RawSimplicialManifold {
  /// Function building the actual mesh data structure
  /// from the raw defining data.
  pub fn build(self) -> Manifold {
    let Self {
      nvertices,
      facets,
      edge_lengths,
    } = self;

    let complex = Complex::from_facets(facets.clone(), nvertices);

    let edges = complex.skeleton(1).keys();
    let edge_lengths =
      na::DVector::from_iterator(edges.len(), edges.map(|edge| edge_lengths[edge]));
    let edge_lengths = EdgeLengths::new(edge_lengths);

    Manifold {
      facets,
      complex,
      edge_lengths,
    }
  }
}
