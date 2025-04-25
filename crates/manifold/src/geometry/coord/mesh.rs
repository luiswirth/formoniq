use super::{
  simplex::{SimplexCoords, SimplexHandleExt},
  Coord, CoordRef,
};
use crate::{
  geometry::metric::mesh::MeshLengths,
  topology::{complex::Complex, handle::SimplexHandle, VertexIdx},
  Dim,
};

use common::linalg::nalgebra::{Matrix, Vector};

use itertools::Itertools;

/// The coordinates of the vertices of the mesh.
#[derive(Debug, Clone)]
pub struct MeshCoords {
  matrix: Matrix,
}

impl MeshCoords {
  pub fn standard(ndim: Dim) -> Self {
    SimplexCoords::standard(ndim).vertices
  }
  pub fn new(matrix: Matrix) -> Self {
    Self { matrix }
  }

  pub fn matrix(&self) -> &Matrix {
    &self.matrix
  }
  pub fn matrix_mut(&mut self) -> &mut Matrix {
    &mut self.matrix
  }
  pub fn into_matrix(self) -> Matrix {
    self.matrix
  }

  pub fn swap_coords(&mut self, icol: usize, jcol: usize) {
    self.matrix.swap_columns(icol, jcol)
  }
}

impl From<Matrix> for MeshCoords {
  fn from(matrix: Matrix) -> Self {
    Self::new(matrix)
  }
}

impl From<&[Coord]> for MeshCoords {
  fn from(vectors: &[Coord]) -> Self {
    let matrix = Matrix::from_columns(vectors);
    Self::new(matrix)
  }
}

impl MeshCoords {
  pub fn dim(&self) -> Dim {
    self.matrix.nrows()
  }
  pub fn nvertices(&self) -> usize {
    self.matrix.ncols()
  }

  pub fn coord(&self, ivertex: VertexIdx) -> CoordRef {
    self.matrix.column(ivertex)
  }

  pub fn coord_iter(
    &self,
  ) -> na::iter::ColumnIter<'_, f64, na::Dyn, na::Dyn, na::VecStorage<f64, na::Dyn, na::Dyn>> {
    self.matrix.column_iter()
  }

  pub fn coord_iter_mut(
    &mut self,
  ) -> na::iter::ColumnIterMut<'_, f64, na::Dyn, na::Dyn, na::VecStorage<f64, na::Dyn, na::Dyn>> {
    self.matrix.column_iter_mut()
  }

  pub fn to_edge_lengths(&self, topology: &Complex) -> MeshLengths {
    let edges = topology.edges();
    let mut edge_lengths = Vector::zeros(edges.len());
    for (iedge, edge) in edges.handle_iter().enumerate() {
      let [vi, vj] = (*edge).clone().try_into().unwrap();
      let length = (self.coord(vj) - self.coord(vi)).norm();
      edge_lengths[iedge] = length;
    }
    // SAFETY: Edge Lengths come from a coordinate realizations.
    MeshLengths::new_unchecked(edge_lengths)
  }
}

impl MeshCoords {
  pub fn embed_euclidean(mut self, dim: Dim) -> MeshCoords {
    let old_dim = self.matrix.nrows();
    self.matrix = self.matrix.insert_rows(old_dim, dim - old_dim, 0.0);
    self
  }
}

impl MeshCoords {
  pub fn find_cell_containing<'a>(
    &self,
    topology: &'a Complex,
    coord: CoordRef,
  ) -> Option<SimplexHandle<'a>> {
    topology
      .cells()
      .handle_iter()
      .find(|cell| cell.coord_simplex(self).is_global_inside(coord))
  }
}

pub fn standard_coord_complex(dim: Dim) -> (Complex, MeshCoords) {
  let topology = Complex::standard(dim);

  let coords = topology
    .vertices()
    .handle_iter()
    .map(|v| v.kidx())
    .map(|v| {
      let mut vec = Vector::zeros(dim);
      if v > 0 {
        vec[v - 1] = 1.0;
      }
      vec
    })
    .collect_vec();
  let coords = Matrix::from_columns(&coords);
  let coords = MeshCoords::new(coords);

  (topology, coords)
}
