use super::{
  raw::RawSimplicialManifold,
  simplicial::{OrderedVertplex, OrientedVertplex, SimplexExt, Vertplex},
  Manifold,
};
use crate::{combo::Sign, linalg::DMatrixExt as _, mesh::VertexIdx, Dim};

use std::collections::{hash_map, HashMap};

#[derive(Debug, Clone)]
pub struct VertexCoords {
  /// The vertex coordinates in the columns of a matrix.
  matrix: na::DMatrix<f64>,
}
impl VertexCoords {
  pub fn new(matrix: na::DMatrix<f64>) -> Self {
    Self { matrix }
  }

  pub fn dim(&self) -> Dim {
    self.matrix.nrows()
  }
  pub fn nvertices(&self) -> usize {
    self.matrix.ncols()
  }

  pub fn coord(&self, ivertex: VertexIdx) -> na::DVectorView<f64> {
    self.matrix.column(ivertex)
  }

  pub fn matrix(&self) -> &na::DMatrix<f64> {
    &self.matrix
  }
  pub fn matrix_mut(&mut self) -> &mut na::DMatrix<f64> {
    &mut self.matrix
  }
  pub fn into_matrix(self) -> na::DMatrix<f64> {
    self.matrix
  }

  pub fn coord_simplex(&self, simp: &OrderedVertplex) -> CoordSimplex {
    let mut vert_coords = na::DMatrix::zeros(self.dim(), simp.len());
    for (i, &v) in simp.iter().enumerate() {
      vert_coords.set_column(i, &self.coord(v));
    }
    CoordSimplex::new(vert_coords)
  }

  pub fn eval_coord_fn<F>(&self, f: F) -> na::DVector<f64>
  where
    F: FnMut(na::DVectorView<f64>) -> f64,
  {
    na::DVector::from_iterator(self.nvertices(), self.matrix.column_iter().map(f))
  }

  pub fn embed_euclidean(mut self, dim: Dim) -> VertexCoords {
    let old_dim = self.matrix.nrows();
    self.matrix = self.matrix.insert_rows(old_dim, dim - old_dim, 0.0);
    self
  }
}

#[derive(Debug, Clone)]
pub struct CoordManifold {
  /// topology
  cells: Vec<OrientedVertplex>,
  /// geometry
  vertex_coords: VertexCoords,
}
impl CoordManifold {
  pub fn new(cells: Vec<OrientedVertplex>, vertex_coords: VertexCoords) -> Self {
    if cfg!(debug_assertions) {
      let dim_intrinsic = cells[0].dim();
      for cell in &cells {
        assert!(cell.dim() == dim_intrinsic, "Inconsistent cell dimension.");
        let coord_cell = vertex_coords.coord_simplex(&cell.clone().forget_sign());
        assert!(
          coord_cell.orientation() * cell.sign() == Sign::Pos,
          "Cells must be positively oriented."
        );
      }
    }

    Self {
      cells,
      vertex_coords,
    }
  }

  pub fn dim_embedded(&self) -> Dim {
    self.vertex_coords.dim()
  }
  pub fn dim_intrinsic(&self) -> Dim {
    self.cells[0].dim()
  }

  pub fn into_parts(self) -> (Vec<OrientedVertplex>, VertexCoords) {
    (self.cells, self.vertex_coords)
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
          let edge = Vertplex::from([vi, vj]).sort_signed().forget_sign();
          if let hash_map::Entry::Vacant(e) = edge_lengths.entry(edge) {
            let length = (self.vertex_coords.coord(vj) - self.vertex_coords.coord(vi)).norm();
            e.insert(length);
          }
        }
      }
    }

    RawSimplicialManifold::new(self.vertex_coords.nvertices(), self.cells, edge_lengths)
  }

  pub fn into_intrinsic(self) -> Manifold {
    self.into_raw_manifold().build()
  }

  pub fn embed_euclidean(mut self, dim: Dim) -> CoordManifold {
    self.vertex_coords = self.vertex_coords.embed_euclidean(dim);
    self
  }
}

impl CoordManifold {
  pub fn cells(&self) -> &[OrientedVertplex] {
    &self.cells
  }
  pub fn vertex_coords(&self) -> &VertexCoords {
    &self.vertex_coords
  }
  pub fn vertex_coords_mut(&mut self) -> &mut VertexCoords {
    &mut self.vertex_coords
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
