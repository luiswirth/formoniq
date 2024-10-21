//! This module is concerned with the "raw" data required to fully specify and
//! define a coordinate-free simplicial Riemannian manifold.
//! The structs in this module try to be minimal in the sense that they don't
//! include any information that is redundant or can be derived from other fields.
//!
//! A focus is made on separating topological and geometrical data.

use super::{
  Length, ManifoldGeometry, ManifoldTopology, SimplexBetweenVertices, SimplexTopology,
  SimplicialManifold, SkeletonTopology, VertexIdx,
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
      .map(|ivertex| (vec![ivertex], SimplexTopology::stub(vec![ivertex])))
      .collect();

    // add d-simplicies (cells)
    skeletons[dim] = raw
      .topology
      .cells
      .into_iter()
      .map(|c| c.into_vertices())
      .map(|v| {
        let mut sorted = v.clone();
        sorted.sort();
        (sorted, SimplexTopology::stub(v))
      })
      .collect();

    // add all other simplicies in between vertices and cells
    // and store the incidence
    for super_dim in (1..=dim).rev() {
      let sub_dim = super_dim - 1;
      let incidence = &mut incidences[sub_dim];

      let ([.., sub_skel], [super_skel, ..]) = skeletons.split_at_mut(super_dim) else {
        unreachable!()
      };

      for isuper_simp in 0..super_skel.len() {
        let super_simp = super_skel.get_index_mut(isuper_simp).unwrap().1;
        for ivertex in 0..super_simp.vertices.len() {
          let mut rel_orientation = Orientation::from_permutation_parity(ivertex);

          let mut sub_simp = super_simp.vertices.clone();
          sub_simp.remove(ivertex);

          let (isub_simp, sub_simp) = match sub_skel.get_full_mut(&sub_simp) {
            Some((i, _, simp)) => {
              // check if existing simp has different orientation from generated simp
              let mut actual_vertices = simp.vertices.clone();
              let mut this_vertices = sub_simp;
              let actual_swaps = sort_count_swaps(&mut actual_vertices);
              let this_swaps = sort_count_swaps(&mut this_vertices);
              let is_different_orientation = ((actual_swaps - this_swaps) % 2) == 1;
              if is_different_orientation {
                rel_orientation.switch();
              }
              (i, simp)
            }
            None => {
              let sub_simp = SimplexTopology::stub(sub_simp);
              let i = sub_skel.insert_full(sub_simp.vertices.clone(), sub_simp).0;
              let simp = sub_skel.get_index_mut(i).unwrap().1;
              (i, simp)
            }
          };
          super_simp.subs.push((isub_simp, rel_orientation));
          sub_simp.supers.push((isuper_simp, rel_orientation));

          incidence.push((isuper_simp, isub_simp, rel_orientation.as_f64()));
        }
      }
    }

    let edges_between_vertices: HashMap<_, _> = skeletons[1]
      .values()
      .enumerate()
      .map(|(iedge, edge)| {
        let edge = edge.vertices.clone();
        let edge = SimplexBetweenVertices::new(edge);
        (edge, iedge)
      })
      .collect();

    // set edges of simplicies
    #[allow(clippy::needless_range_loop)]
    for skeleton in skeletons.iter_mut().skip(1) {
      for icell in 0..skeleton.len() {
        let vertices = skeleton.get_index(icell).unwrap().1.vertices.clone();
        let nvertices = vertices.len();
        for i in 0..nvertices {
          let vi = vertices[i];
          for j in (i + 1)..nvertices {
            let vj = vertices[j];
            let edge = SimplexBetweenVertices::edge(vi, vj);
            let iedge = edges_between_vertices[&edge];
            skeleton.get_index_mut(icell).unwrap().1.edges.push(iedge);
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

    // finalize skeletons
    let skeletons: Vec<SkeletonTopology> = skeletons
      .into_iter()
      .map(|skel| skel.into_values().collect())
      .collect();

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
