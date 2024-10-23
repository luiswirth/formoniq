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
  topology: RawManifoldTopology,
  /// The data defining the geometric structure of the manifold.
  geometry: RawManifoldGeometry,
}
impl RawSimplicialManifold {
  pub fn new(nnodes: usize, topology: RawManifoldTopology, geometry: RawManifoldGeometry) -> Self {
    Self {
      nnodes,
      topology,
      geometry,
    }
  }
  pub fn nnodes(&self) -> usize {
    self.nnodes
  }
  pub fn dim(&self) -> Dim {
    self.topology.dim()
  }
}

/// The data defining the topological structure of the manifold.
pub struct RawManifoldTopology {
  /// A mapping [`CellIdx`] -> [`RawSimplexTopology`].
  /// Defining the toplogy of the cells and
  /// inducing a global numbering of the cells.
  cells: Vec<RawSimplexTopology>,
}
impl RawManifoldTopology {
  pub fn new(cells: Vec<RawSimplexTopology>) -> Self {
    assert!(!cells.is_empty());

    if cfg!(debug_assertions) {
      let dim = cells[0].dim();
      for c in &cells {
        debug_assert_eq!(c.dim(), dim);
      }
    }
    Self { cells }
  }

  pub fn dim(&self) -> Dim {
    self.cells[0].dim()
  }
}

/// The data defining the topological structure of a simplex.
/// The only relevant information is which vertices compose the simplex.
#[derive(Clone)]
pub struct RawSimplexTopology {
  /// The indices of the vertices that form the simplex.
  pub vertices: Vec<VertexIdx>,
}
impl RawSimplexTopology {
  pub fn new(vertices: Vec<VertexIdx>) -> Self {
    Self { vertices }
  }
  pub fn vertices(&self) -> &[VertexIdx] {
    &self.vertices
  }
  pub fn nvertices(&self) -> usize {
    self.vertices.len()
  }
  pub fn dim(&self) -> Dim {
    self.vertices().len() - 1
  }
  pub fn into_vertices(self) -> Vec<VertexIdx> {
    self.vertices
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

    for (icell, cell) in raw.topology.cells.into_iter().enumerate() {
      for (sub_dim, subs) in skeletons.iter_mut().enumerate() {
        for sub in cell.vertices.iter().copied().combinations(sub_dim + 1) {
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
      let length = raw.geometry.edge_lengths[&edge];
      edge_lengths.push(length);
    }

    Self {
      topology: ManifoldTopology { skeletons },
      geometry: ManifoldGeometry { edge_lengths },
    }
  }
}
