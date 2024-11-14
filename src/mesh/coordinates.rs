use super::{raw::RawSimplicialManifold, SimplicialManifold};
use crate::{
  combinatorics::{OrderedSimplex, Orientation, OrientedSimplex, SortedSimplex},
  mesh::VertexIdx,
  Dim,
};

use std::collections::{hash_map, HashMap};

#[derive(Debug, Clone)]
pub struct NodeCoords {
  /// The coordinates of the nodes in the columns of a matrix.
  coords: na::DMatrix<f64>,
}
impl NodeCoords {
  pub fn new(coords: na::DMatrix<f64>) -> Self {
    Self { coords }
  }
  pub fn dim(&self) -> Dim {
    self.coords.nrows()
  }
  pub fn nnodes(&self) -> usize {
    self.coords.ncols()
  }
  pub fn coords(&self) -> &na::DMatrix<f64> {
    &self.coords
  }
  pub fn coord(&self, inode: VertexIdx) -> na::DVectorView<f64> {
    self.coords.column(inode)
  }

  pub fn coord_simplex(&self, simp: &OrderedSimplex) -> CoordSimplex {
    let mut vert_coords = na::DMatrix::zeros(simp.dim(), simp.nvertices());
    for (i, &v) in simp.iter().enumerate() {
      vert_coords.set_column(i, &self.coord(v));
    }
    CoordSimplex::new(vert_coords)
  }
}

pub struct CoordManifold {
  /// topology
  cells: Vec<OrientedSimplex>,
  /// geometry
  node_coords: NodeCoords,
}
impl CoordManifold {
  pub fn new(cells: Vec<OrientedSimplex>, node_coords: NodeCoords) -> Self {
    if cfg!(debug_assertions) {
      for cell in &cells {
        let coord_cell = node_coords.coord_simplex(cell.ordered());
        debug_assert!(
          coord_cell.orientation() * cell.orientation() == Orientation::Pos,
          "Cells must be positively oriented."
        );
      }
    }

    Self { cells, node_coords }
  }

  pub fn into_raw_manifold(self) -> RawSimplicialManifold {
    let mut edge_lengths = HashMap::new();

    // TODO: can we optimize this and avoid iterating over all cells?
    // this would require already knowing all edges
    for cell in &self.cells {
      for i in 0..cell.nvertices() {
        let vi = cell[i];
        for j in (i + 1)..cell.nvertices() {
          let vj = cell[j];
          let edge = SortedSimplex::edge(vi, vj);
          if let hash_map::Entry::Vacant(e) = edge_lengths.entry(edge) {
            let length = (self.node_coords.coord(vj) - self.node_coords.coord(vi)).norm();
            e.insert(length);
          }
        }
      }
    }

    RawSimplicialManifold::new(self.node_coords.nnodes(), self.cells, edge_lengths)
  }

  pub fn into_manifold(self) -> SimplicialManifold {
    SimplicialManifold::new(self.into_raw_manifold())
  }
}

impl CoordManifold {
  pub fn cells(&self) -> &[OrientedSimplex] {
    &self.cells
  }
  pub fn node_coords(&self) -> &NodeCoords {
    &self.node_coords
  }
}

pub struct CoordSimplex {
  vertices: na::DMatrix<f64>,
}
impl CoordSimplex {
  pub fn new(vertices: na::DMatrix<f64>) -> Self {
    Self { vertices }
  }
}
impl CoordSimplex {
  pub fn nvertices(&self) -> usize {
    self.vertices.ncols()
  }
  pub fn dim(&self) -> Dim {
    self.nvertices() - 1
  }

  pub fn spanning_vectors(&self) -> na::DMatrix<f64> {
    let dim = self.dim();
    let mut mat = na::DMatrix::zeros(dim, dim);
    let v0 = self.vertices.column(0);
    for (i, vi) in self.vertices.column_iter().skip(1).enumerate() {
      let v0i = vi - v0;
      mat.set_column(i, &v0i);
    }
    mat
  }
  pub fn det(&self) -> f64 {
    self.spanning_vectors().determinant()
  }
  pub fn vol(&self) -> f64 {
    self.det().abs()
  }
  pub fn orientation(&self) -> Orientation {
    Orientation::from_det(self.det())
  }
}
