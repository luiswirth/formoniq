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

impl RawSimplicialManifold {
  /// Function building the actual mesh data structure
  /// from the raw defining data.
  pub fn build(self) -> SimplicialManifold {
    let dim = self.dim();

    let cells = self.cells;

    let mut skeletons = vec![Skeleton::new(); dim + 1];
    skeletons[0] = (0..self.nnodes)
      .map(|v| (CanonicalVertplex::vertex(v), SimplexData::stub()))
      .collect();

    for (icell, cell) in cells.iter().enumerate() {
      let cell = cell.clone().into_canonical();
      for (sub_dim, subs) in skeletons.iter_mut().enumerate() {
        for sub in cell.subs(sub_dim) {
          let sub = subs.entry(sub.clone()).or_insert(SimplexData::stub());
          sub.parent_cells.push(icell);
        }
      }
    }

    // Assert consistent orientation of cells.
    // Two adjacent cells are consistently oriented if their shared facet appears
    // with opposite orientations from each cell's perspective.
    let facets = &skeletons[dim - 1];
    for (facet, SimplexData { parent_cells }) in facets {
      let parent_cells = parent_cells.iter().map(|&cell_kidx| &cells[cell_kidx]);

      // The same facet, but as seen from the 1 or 2 adjacent cells.
      let cells_facet = parent_cells
        .map(|cell| {
          cell
            .boundary()
            .into_iter()
            .find(|b| b.as_canonical() == facet)
            .unwrap()
        })
        .collect::<Vec<_>>();

      let is_boundary_facet = cells_facet.len() == 1;
      if !is_boundary_facet {
        assert!(cells_facet.len() == 2);
        assert!(
          !cells_facet[0].orientation_eq(&cells_facet[1]).unwrap(),
          "Manifold cells must be consistently oriented."
        );
      }
    }

    let mut edge_lengths = Vec::new();
    let edges = skeletons[1].keys();
    for edge in edges {
      let length = self.edge_lengths[edge];
      edge_lengths.push(length);
    }

    SimplicialManifold {
      cells,
      skeletons,
      edge_lengths,
    }
  }
}
