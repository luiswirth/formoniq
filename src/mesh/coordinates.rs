use super::{raw::RawSimplicialManifold, SimplicialManifold};
use crate::{
  combo::{
    simplicial::{OrderedVertplex, OrientedVertplex, SimplexExt as _, Vertplex},
    Sign,
  },
  linalg::DMatrixExt as _,
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
  pub fn into_inner(self) -> na::DMatrix<f64> {
    self.coords
  }

  pub fn eval_coord_fn<F>(&self, f: F) -> na::DVector<f64>
  where
    F: FnMut(na::DVectorView<f64>) -> f64,
  {
    na::DVector::from_iterator(self.nnodes(), self.coords.column_iter().map(f))
  }

  pub fn coord_simplex(&self, simp: &OrderedVertplex) -> CoordSimplex {
    let mut vert_coords = na::DMatrix::zeros(self.dim(), simp.len());
    for (i, &v) in simp.iter().enumerate() {
      vert_coords.set_column(i, &self.coord(v));
    }
    CoordSimplex::new(vert_coords)
  }

  pub fn embed_flat(mut self, dim: usize) -> NodeCoords {
    let old_dim = self.coords.nrows();
    self.coords = self.coords.insert_rows(old_dim, dim - old_dim, 0.0);
    self
  }
}

#[derive(Debug, Clone)]
pub struct CoordManifold {
  /// topology
  cells: Vec<OrientedVertplex>,
  /// geometry
  node_coords: NodeCoords,
}
impl CoordManifold {
  pub fn new(cells: Vec<OrientedVertplex>, node_coords: NodeCoords) -> Self {
    if cfg!(debug_assertions) {
      let dim_intrinsic = cells[0].dim();
      for cell in &cells {
        assert!(cell.dim() == dim_intrinsic, "Inconsistent cell dimension.");
        let coord_cell = node_coords.coord_simplex(&cell.clone().forget_sign());
        assert!(
          coord_cell.orientation() * cell.sign() == Sign::Pos,
          "Cells must be positively oriented."
        );
      }
    }

    Self { cells, node_coords }
  }

  pub fn dim_embedded(&self) -> Dim {
    self.node_coords.dim()
  }
  pub fn dim_intrinsic(&self) -> Dim {
    self.cells[0].dim()
  }

  pub fn into_parts(self) -> (Vec<OrientedVertplex>, NodeCoords) {
    (self.cells, self.node_coords)
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
          let edge = Vertplex::from([vi, vj]).into_sorted().forget_sign();
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
    self.into_raw_manifold().build()
  }

  pub fn embed_flat(mut self, dim: Dim) -> CoordManifold {
    self.node_coords = self.node_coords.embed_flat(dim);
    self
  }
}

impl CoordManifold {
  pub fn cells(&self) -> &[OrientedVertplex] {
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
  pub fn dim_intrinsic(&self) -> Dim {
    self.nvertices() - 1
  }
  pub fn dim_embedded(&self) -> Dim {
    self.vertices.nrows()
  }

  // TODO: is this a good name?
  pub fn is_euclidean(&self) -> bool {
    self.dim_intrinsic() == self.dim_embedded()
  }

  pub fn spanning_vectors(&self) -> na::DMatrix<f64> {
    let mut mat = na::DMatrix::zeros(self.dim_embedded(), self.dim_intrinsic());
    let v0 = self.vertices.column(0);
    for (i, vi) in self.vertices.column_iter().skip(1).enumerate() {
      let v0i = vi - v0;
      mat.set_column(i, &v0i);
    }
    mat
  }
  pub fn det(&self) -> f64 {
    if self.is_euclidean() {
      self.spanning_vectors().determinant()
    } else {
      self.spanning_vectors().gram_det_sqrt()
    }
  }
  pub fn vol(&self) -> f64 {
    self.det().abs()
  }
  pub fn orientation(&self) -> Sign {
    Sign::from_f64(self.det())
  }
}
