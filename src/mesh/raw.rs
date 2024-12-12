//! Module for creating simplicial manifolds
//!
//! This module is concerned with the "raw" data required to fully specify and
//! define a coordinate-free simplicial Riemannian manifold.
//! This includes both topological and geometric data.
//! The structs in this module try to be minimal in the sense that they don't
//! include any information that is redundant or can be derived from other fields.

use super::{SimplexData, SimplicialManifold, Skeleton};
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
  cells: Vec<OrientedVertplex>,
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
      cells,
      edge_lengths,
    }
  }
  pub fn dim(&self) -> Dim {
    self.cells[0].dim()
  }
  pub fn nvertices(&self) -> usize {
    self.nvertices
  }
}

impl RawSimplicialManifold {
  /// Function building the actual mesh data structure
  /// from the raw defining data.
  pub fn build(self) -> SimplicialManifold {
    let dim = self.dim();

    let cells = self.cells;

    let mut skeletons = vec![Skeleton::new(); dim + 1];
    skeletons[0] = (0..self.nvertices)
      .map(|v| (SortedVertplex::single(v), SimplexData::stub()))
      .collect();

    for (icell, cell) in cells.iter().enumerate() {
      let cell = cell.clone().into_sorted();
      for (dim_sub, subs) in skeletons.iter_mut().enumerate() {
        let nvertices_sub = dim_sub + 1;
        for sub in cell.subs(nvertices_sub) {
          let sub = subs.entry(sub.clone()).or_insert(SimplexData::stub());
          sub.parent_cells.push(icell);
        }
      }
    }

    // Assert consistent orientation of cells.
    // Two adjacent cells are consistently oriented if their shared face appears
    // with opposite orientations from each cell's perspective.
    let faces = &skeletons[dim - 1];
    for (face, SimplexData { parent_cells }) in faces {
      let parent_cells = parent_cells.iter().map(|&cell_kidx| &cells[cell_kidx]);

      // The same face, but as seen from the 1 or 2 adjacent cells.
      let cells_face = parent_cells
        .map(|cell| {
          cell
            .boundary()
            .find(|b| b.clone().into_sorted().forget_sign() == *face)
            .unwrap()
        })
        .collect::<Vec<_>>();

      let is_boundary_face = cells_face.len() == 1;
      if !is_boundary_face {
        assert_eq!(
          cells_face.len(),
          2,
          "Non-manifold topology at face {face:?}."
        );
        assert!(
          !cells_face[0].orientation_eq(&cells_face[1]),
          "Manifold cells must be consistently oriented."
        );
      }
    }

    let edges = skeletons[1].keys();
    let edge_lengths =
      na::DVector::from_iterator(edges.len(), edges.map(|edge| self.edge_lengths[edge]));
    let edge_lengths = EdgeLengths::new(edge_lengths);

    SimplicialManifold {
      cells,
      skeletons,
      edge_lengths,
    }
  }
}
