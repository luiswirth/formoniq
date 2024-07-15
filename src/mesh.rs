//! This module defined the objects necessary for a Mesh.
//! A mesh plays the role of a container of mesh entities (cells,faces,edges,vertices).
//! It must allows the unique identification of the entities.
//! It must allow from traversal of the cells -> must induce a global numbering on all entities.
//! It must represent mesh topology (incidence).
//! It must represent mesh geometry (location, shape)

pub mod factory;
pub mod gmsh;
pub mod hypercube;

use crate::{geometry::CoordSimplex, Dim};

pub type NodeId = usize;
pub type CellId = usize;

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
pub struct SimplicialMesh {
  node_coords: na::DMatrix<f64>,
  cells: Vec<MeshSimplex>,
}
impl SimplicialMesh {
  pub fn new(node_coords: na::DMatrix<f64>, cells: Vec<MeshSimplex>) -> Self {
    Self { node_coords, cells }
  }

  pub fn dim_intrinsic(&self) -> Dim {
    self.cells.len() - 1
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

  pub fn ncells(&self) -> usize {
    self.cells.len()
  }
  /// cells of the mesh
  pub fn cells(&self) -> &[MeshSimplex] {
    &self.cells
  }
  pub fn cell(&self, id: CellId) -> &MeshSimplex {
    &self.cells[id]
  }

  pub fn coordinate_cell(&self, id: CellId) -> CoordSimplex {
    let cell = &self.cells[id];
    let mut vertices = na::DMatrix::zeros(self.dim_ambient(), cell.nvertices());
    for (i, &v) in cell.vertices().iter().enumerate() {
      vertices
        .column_mut(i)
        .copy_from(&self.node_coords.column(v));
    }
    CoordSimplex::new(vertices)
  }

  pub fn mesh_width(&self) -> f64 {
    (0..self.cells().len())
      .map(|icell| self.coordinate_cell(icell).diameter())
      .max_by(|a, b| a.partial_cmp(b).unwrap())
      .unwrap()
  }

  pub fn shape_regularity_measure(&self) -> f64 {
    (0..self.cells().len())
      .map(|icell| self.coordinate_cell(icell).shape_reguarity_measure())
      .max_by(|a, b| a.partial_cmp(b).unwrap())
      .unwrap()
  }
}
