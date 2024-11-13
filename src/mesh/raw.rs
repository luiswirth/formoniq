//! Module for creating simplicial manifolds
//!
//! This module is concerned with the "raw" data required to fully specify and
//! define a coordinate-free simplicial Riemannian manifold.
//! This includes both topological and geometric data.
//! The structs in this module try to be minimal in the sense that they don't
//! include any information that is redundant or can be derived from other fields.

use super::{
  Length, ManifoldGeometry, ManifoldSimplex, ManifoldTopology, SimplicialManifold, SortedSimplex,
};
use crate::{combinatorics::OrientedSimplex, Dim};

use indexmap::IndexMap;
use std::collections::HashMap;

/// The data defining a simplicial Riemanninan manifold.
pub struct RawSimplicialManifold {
  /// number of nodes
  nnodes: usize,
  /// A mapping [`CellIdx`] -> [`RawSimplexTopology`].
  /// Defines topology (connectivity + orientation) and global numbering/order of cells.
  cells: Vec<OrientedSimplex>,
  /// A mapping [`SortedSimplex`] -> [`Length`].
  /// Defines geometry of the manifold through the lengths of all edges.
  edge_lengths: HashMap<SortedSimplex, Length>,
}
impl RawSimplicialManifold {
  pub fn new(
    nnodes: usize,
    cells: Vec<OrientedSimplex>,
    edge_lengths: HashMap<SortedSimplex, Length>,
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

    let mut skeletons = vec![IndexMap::new(); dim + 1];
    skeletons[0] = (0..raw.nnodes)
      .map(|v| {
        (
          SortedSimplex::vertex(v),
          ManifoldSimplex::new(OrientedSimplex::vertex(v), Vec::new()),
        )
      })
      .collect();

    for (icell, cell) in raw.cells.into_iter().enumerate() {
      for (sub_dim, subs) in skeletons.iter_mut().enumerate() {
        // TODO: this shouldn't be boundary but `cell.sub(sub_dim)`.
        panic!();
        for sub in cell.boundary() {
          let sorted = SortedSimplex::from(sub.ordered().clone());
          let sub = subs
            .entry(sorted)
            .and_modify(|existent_sub| {
              // Assert consistent orientation of cells.
              // The orientation is consistent if a shared face is oriented
              // opposite as viewed from the two adjacent cells.
              let is_face = sub_dim == dim - 1;
              if is_face {
                assert!(!existent_sub.vertices.orientation_eq(&sub).unwrap());
              }
            })
            .or_insert(ManifoldSimplex::new(sub, Vec::new()));
          sub.cells.push(icell);
        }
      }
    }

    dbg!(&skeletons);

    // set edge lengths of mesh
    let mut edge_lengths = Vec::new();
    for edge in skeletons[1].values() {
      let edge = edge.vertices().sorted();
      let length = raw.edge_lengths[edge];
      edge_lengths.push(length);
    }

    Self {
      topology: ManifoldTopology { skeletons },
      geometry: ManifoldGeometry { edge_lengths },
    }
  }
}
