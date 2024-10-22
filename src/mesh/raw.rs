//! This module is concerned with the "raw" data required to fully specify and
//! define a coordinate-free simplicial Riemannian manifold.
//! The structs in this module try to be minimal in the sense that they don't
//! include any information that is redundant or can be derived from other fields.
//!
//! A focus is made on separating topological and geometrical data.

use super::{
  Length, ManifoldGeometry, ManifoldTopology, SimplexBetweenVertices, SimplexTopology,
  SimplicialManifold, VertexIdx,
};
use crate::{combinatorics::sort_count_swaps, matrix::SparseMatrix, Dim, Orientation};

use indexmap::IndexMap;
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
  /// A mapping [`EdgeBetweenVertices`] -> [`Length`]
  /// Defining the lengths of all edges of the manifold.
  pub edge_lengths: HashMap<SimplexBetweenVertices, Length>,
}
impl RawManifoldGeometry {
  pub fn new(edge_lengths: HashMap<SimplexBetweenVertices, Length>) -> Self {
    Self { edge_lengths }
  }
  pub fn into_edge_lengths(self) -> HashMap<SimplexBetweenVertices, Length> {
    self.edge_lengths
  }
}

impl SimplicialManifold {
  /// Function building the actual mesh data structure
  /// from the raw defining data.
  pub fn from_raw(raw: RawSimplicialManifold) -> Self {
    let dim = raw.dim();
    let nnodes = raw.nnodes();

    let mut skeletons = vec![IndexMap::new(); dim + 1];
    let mut incidences = vec![Vec::new(); dim];

    // add 0-simplicies (vertices)
    skeletons[0] = (0..nnodes)
      .map(|ivertex| {
        (
          SimplexBetweenVertices::vertex(ivertex),
          SimplexTopology::stub(vec![ivertex]),
        )
      })
      .collect();

    // add d-simplicies (cells)
    skeletons[dim] = raw
      .topology
      .cells
      .into_iter()
      .map(|c| c.into_vertices())
      .map(|v| {
        (
          SimplexBetweenVertices::new(v.clone()),
          SimplexTopology::stub(v),
        )
      })
      .collect();

    // add all other simplicies in between vertices and cells
    // and store the incidence
    for super_dim in (1..=dim).rev() {
      let sub_dim = super_dim - 1;
      let incidence = &mut incidences[sub_dim];

      let ([.., subs], [sups, ..]) = skeletons.split_at_mut(super_dim) else {
        unreachable!()
      };

      for isup in 0..sups.len() {
        let (_, sup) = sups.get_index_mut(isup).unwrap();
        for ivertex in 0..sup.vertices.len() {
          let mut rel_orientation = Orientation::from_permutation_parity(ivertex);

          let mut sub = sup.vertices.clone();
          sub.remove(ivertex);
          let sub_sorted = SimplexBetweenVertices::new(sub.clone());

          let (isub, sub) = match subs.get_full_mut(&sub_sorted) {
            Some((existing_isub, _, existing_sub)) => {
              // TODO: make this orientation code more intuitive
              // check if existing simp has different orientation from generated simp
              let mut actual_vertices = existing_sub.vertices.clone();
              let mut this_vertices = sub.clone();
              let actual_swaps = sort_count_swaps(&mut actual_vertices);
              let this_swaps = sort_count_swaps(&mut this_vertices);
              let is_different_orientation = ((actual_swaps - this_swaps) % 2) == 1;
              if is_different_orientation {
                rel_orientation.switch();
              }
              (existing_isub, existing_sub)
            }
            None => {
              let sub = SimplexTopology::stub(sub);
              let i = subs.insert_full(sub_sorted, sub).0;
              let sub = subs.get_index_mut(i).unwrap().1;
              (i, sub)
            }
          };
          sup.subs.push((isub, rel_orientation));
          sub.supers.push((isup, rel_orientation));

          incidence.push((isup, isub, rel_orientation.as_f64()));
        }
      }
    }

    // set edges of simplicies
    #[allow(clippy::needless_range_loop)]
    for d in 1..=dim {
      for icell in 0..skeletons[d].len() {
        let vertices = skeletons[d].get_index(icell).unwrap().1.vertices.clone();
        let nvertices = vertices.len();
        for i in 0..nvertices {
          let vi = vertices[i];
          for j in (i + 1)..nvertices {
            let vj = vertices[j];
            let edge = SimplexBetweenVertices::edge(vi, vj);
            let iedge = skeletons[1].get_full(&edge).unwrap().0;
            skeletons[d]
              .get_index_mut(icell)
              .unwrap()
              .1
              .edges
              .push(iedge);
          }
        }
      }
    }

    // set edge lengths of mesh
    let mut edge_lengths = Vec::new();
    for edge in skeletons[1].values() {
      let edge = edge.vertices.clone();
      let edge = SimplexBetweenVertices::new(edge);
      let length = raw.geometry.edge_lengths[&edge];
      edge_lengths.push(length);
    }

    // finalize incidence matrices
    let incidence_matrices = incidences
      .into_iter()
      .enumerate()
      .map(|(k, triplets)| {
        SparseMatrix::from_triplets(skeletons[k + 1].len(), skeletons[k].len(), triplets)
      })
      .collect();

    Self {
      topology: ManifoldTopology {
        skeletons,
        incidence_matrices,
      },
      geometry: ManifoldGeometry { edge_lengths },
    }
  }
}
