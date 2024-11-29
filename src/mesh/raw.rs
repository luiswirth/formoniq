//! Module for creating simplicial manifolds
//!
//! This module is concerned with the "raw" data required to fully specify and
//! define a coordinate-free simplicial Riemannian manifold.
//! This includes both topological and geometric data.
//! The structs in this module try to be minimal in the sense that they don't
//! include any information that is redundant or can be derived from other fields.

use super::{Length, SimplexData, SimplicialManifold, Skeleton};
use crate::{
  combinatorics::{CanonicalVertplex, OrientedVertplex},
  Dim,
};

use std::collections::HashMap;

/// The data defining a simplicial Riemanninan manifold.
pub struct RawSimplicialManifold {
  /// number of nodes
  nnodes: usize,
  /// A mapping [`CellIdx`] -> [`RawSimplexTopology`].
  /// Defines topology (connectivity + orientation) and global numbering/order of cells.
  cells: Vec<OrientedVertplex>,
  /// A mapping [`SortedSimplex`] -> [`Length`].
  /// Defines geometry of the manifold through the lengths of all edges.
  edge_lengths: HashMap<CanonicalVertplex, Length>,
}
impl RawSimplicialManifold {
  pub fn new(
    nnodes: usize,
    cells: Vec<OrientedVertplex>,
    edge_lengths: HashMap<CanonicalVertplex, Length>,
  ) -> Self {
    Self {
      nnodes,
      cells,
      edge_lengths,
    }
  }
  pub fn dim(&self) -> Dim {
    self.cells[0].dim()
  }
  pub fn nnodes(&self) -> usize {
    self.nnodes
  }
}

impl SimplicialManifold {
  /// Function building the actual mesh data structure
  /// from the raw defining data.
  pub fn new(raw: RawSimplicialManifold) -> Self {
    let dim = raw.dim();

    let cells = raw.cells;

    let mut skeletons = vec![Skeleton::new(); dim + 1];
    skeletons[0].simplicies = (0..raw.nnodes)
      .map(|v| (CanonicalVertplex::vertex(v), SimplexData::stub()))
      .collect();

    for (icell, cell) in cells.iter().enumerate() {
      let cell = cell.clone().into_canonical();
      for (sub_dim, Skeleton { simplicies: subs }) in skeletons.iter_mut().enumerate() {
        for sub in cell.subs(sub_dim) {
          let sub = subs.entry(sub.clone()).or_insert(SimplexData::stub());
          sub.parent_cells.push(icell);
        }
      }
    }

    // Assert consistent orientation of cells.
    // The orientation of two adjacent cells is consistent if the shared facet
    // has opposing orientations.
    for (facet, facet_data) in &skeletons[dim - 1].simplicies {
      let oriented_facets = &facet_data
        .parent_cells
        .iter()
        .map(|&cell_kidx| &cells[cell_kidx])
        .map(|cell| {
          cell
            .boundary()
            .into_iter()
            .find(|b| b.as_canonical() == facet)
            .unwrap()
        })
        .collect::<Vec<_>>();

      let is_boundary_facet = oriented_facets.len() == 1;
      if is_boundary_facet {
        continue;
      }

      assert!(
        oriented_facets.len() == 2,
        "Each cell has exactly two facets."
      );
      assert!(
        !oriented_facets[0]
          .orientation_eq(&oriented_facets[1])
          .unwrap(),
        "Manifold cells must be consistently oriented."
      );
    }
    //if is_face {
    //assert!(!existent_sub.sorted_vertices.orientation_eq(&sub).unwrap());
    //}

    // set edge lengths of mesh
    let mut edge_lengths = Vec::new();
    for edge in skeletons[1].simplicies.keys() {
      let length = raw.edge_lengths[edge];
      edge_lengths.push(length);
    }

    Self {
      cells,
      skeletons,
      edge_lengths,
    }
  }
}
