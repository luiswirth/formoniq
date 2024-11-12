//! Module for creating simplicial manifolds
//!
//! This module is concerned with the "raw" data required to fully specify and
//! define a coordinate-free simplicial Riemannian manifold.
//! The structs in this module try to be minimal in the sense that they don't
//! include any information that is redundant or can be derived from other fields.

use super::{
  Length, ManifoldGeometry, ManifoldTopology, SimplexTopology, SimplicialManifold, SortedSimplex,
  VertexIdx,
};
use crate::Dim;

use indexmap::IndexMap;
use itertools::Itertools;
use std::collections::HashMap;

/// The data defining a simplicial Riemanninan manifold.
pub struct RawSimplicialManifold {
  nnodes: usize,
  /// The data defining the topological structure of the manifold.
  /// A mapping [`CellIdx`] -> [`RawSimplexTopology`].
  /// Defines connectivity, orientation and global numbering of cells.
  cells: Vec<SimplexVertices>,
  /// The data defining the geometric structure of the manifold.
  /// Defining the lengths of all edges of the manifold.
  edge_lengths: HashMap<SortedSimplex, Length>,
}
impl RawSimplicialManifold {
  pub fn new(
    nnodes: usize,
    cells: Vec<SimplexVertices>,
    edge_lengths: HashMap<SortedSimplex, Length>,
  ) -> Self {
    Self {
      nnodes,
      cells,
      edge_lengths,
    }
  }
  pub fn nnodes(&self) -> usize {
    self.nnodes
  }
  pub fn dim(&self) -> Dim {
    self.cells[0].dim()
  }
}

/// The data defining the topological structure of a simplex.
/// The only relevant information is which vertices compose the simplex.
#[derive(Debug, Clone)]
pub struct SimplexVertices(pub Vec<VertexIdx>);
impl SimplexVertices {
  pub fn new(vertices: Vec<VertexIdx>) -> Self {
    Self(vertices)
  }
  pub fn nvertices(&self) -> usize {
    self.0.len()
  }
  pub fn dim(&self) -> Dim {
    self.0.len() - 1
  }
}
impl std::ops::Deref for SimplexVertices {
  type Target = Vec<VertexIdx>;
  fn deref(&self) -> &Self::Target {
    &self.0
  }
}
impl std::ops::DerefMut for SimplexVertices {
  fn deref_mut(&mut self) -> &mut Self::Target {
    &mut self.0
  }
}

/// The data defining the geometric structure of the manifold.
pub struct RawManifoldGeometry {
  /// Defining the lengths of all edges of the manifold.
  pub edge_lengths: HashMap<SortedSimplex, Length>,
}
impl RawManifoldGeometry {
  pub fn new(edge_lengths: HashMap<SortedSimplex, Length>) -> Self {
    Self { edge_lengths }
  }
  pub fn into_edge_lengths(self) -> HashMap<SortedSimplex, Length> {
    self.edge_lengths
  }
}

impl SimplicialManifold {
  /// Function building the actual mesh data structure
  /// from the raw defining data.
  pub fn from_raw(raw: RawSimplicialManifold) -> Self {
    let dim = raw.dim();

    let mut skeletons = vec![IndexMap::new(); dim + 1];
    skeletons[0] = (0..raw.nnodes)
      .map(|v| {
        (
          SortedSimplex::vertex(v),
          SimplexTopology::new(vec![v], Vec::new()),
        )
      })
      .collect();

    for (icell, cell) in raw.cells.into_iter().enumerate() {
      for (sub_dim, subs) in skeletons.iter_mut().enumerate() {
        for sub in cell.iter().copied().combinations(sub_dim + 1) {
          let sorted = SortedSimplex::new(sub.clone());
          let sub = subs
            .entry(sorted)
            .or_insert(SimplexTopology::new(sub, Vec::new()));
          sub.cells.push(icell);
        }
      }
    }

    // set edge lengths of mesh
    let mut edge_lengths = Vec::new();
    for edge in skeletons[1].values() {
      let edge = edge.vertices.clone();
      let edge = SortedSimplex::new(edge);
      let length = raw.edge_lengths[&edge];
      edge_lengths.push(length);
    }

    Self {
      topology: ManifoldTopology { skeletons },
      geometry: ManifoldGeometry { edge_lengths },
    }
  }
}
