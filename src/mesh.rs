//! This module defined the objects necessary for a Mesh.
//! A mesh plays the role of a container of mesh entities (cells,faces,edges,vertices).
//! It must allows the unique identification of the entities.
//! It must allow from traversal of the cells -> must induce a global numbering on all entities.
//! It must represent mesh topology (incidence).
//! It must represent mesh geometry (location, shape)

pub mod factory;
pub mod gmsh;

use crate::{geometry::CoordSimplex, Dim};

pub type NodeId = usize;
pub type EntityId = (Dim, usize);

/// This is what is called an Entity in LehrFEMpp
#[derive(Debug, Clone)]
pub struct MeshSimplex {
  vertices: Vec<NodeId>,
  sorted_vertices: Vec<NodeId>,
}

impl MeshSimplex {
  pub fn new(vertices: Vec<NodeId>) -> Self {
    assert!(!vertices.is_empty(), "vertices cannot be empty");
    let mut sorted_vertices = vertices.clone();
    sorted_vertices.sort_unstable();

    Self {
      vertices,
      sorted_vertices,
    }
  }
  pub fn dim_intrinsic(&self) -> Dim {
    self.vertices.len() - 1
  }
  pub fn nvertices(&self) -> usize {
    self.vertices.len()
  }
  pub fn vertices(&self) -> &[NodeId] {
    &self.vertices
  }
}
// NOTE: this doesn't care about orientation
impl PartialEq for MeshSimplex {
  fn eq(&self, other: &Self) -> bool {
    self.sorted_vertices == other.sorted_vertices
  }
}
impl Eq for MeshSimplex {}

/// A Simplicial Mesh or Triangulation.
#[derive(Debug)]
pub struct Mesh {
  node_coords: na::DMatrix<f64>,
  simplicies: Vec<Vec<MeshSimplex>>,
  face_relation: Vec<Vec<Vec<usize>>>,
}
impl Mesh {
  pub fn dim_intrinsic(&self) -> Dim {
    self.simplicies.len() - 1
  }
  pub fn dim_ambient(&self) -> Dim {
    self.node_coords.nrows()
  }
  pub fn nnodes(&self) -> usize {
    self.node_coords.ncols()
  }
  /// The dimension of highest dimensional [`Skeleton`]
  pub fn node_coords(&self) -> &na::DMatrix<f64> {
    &self.node_coords
  }
  pub fn simplex_by_id(&self, id: EntityId) -> &MeshSimplex {
    &self.simplicies[id.0][id.1]
  }
  pub fn simplicies(&self) -> &[Vec<MeshSimplex>] {
    &self.simplicies
  }
  pub fn dsimplicies(&self, d: Dim) -> &[MeshSimplex] {
    &self.simplicies[d]
  }
  pub fn coordinate_simplex(&self, id: EntityId) -> CoordSimplex {
    let entity = self.simplex_by_id(id);
    let mut vertices = na::DMatrix::zeros(self.dim_ambient(), entity.nvertices());
    for (i, &v) in entity.vertices().iter().enumerate() {
      vertices
        .column_mut(i)
        .copy_from(&self.node_coords.column(v));
    }
    CoordSimplex::new(vertices)
  }
  /// returns the faces of the given entity
  pub fn simplex_faces(&self, id: EntityId) -> &[usize] {
    if id.0 == 0 {
      &[]
    } else {
      &self.face_relation[id.0 - 1][id.1]
    }
  }
}
