pub mod local;
pub mod quadrature;

use local::SimplexCoords;

use crate::{
  geometry::metric::MeshEdgeLengths,
  topology::{complex::Complex, VertexIdx},
  Dim,
};

use itertools::Itertools;

pub type Coord = na::DVector<f64>;
pub type CoordRef<'a> = na::DVectorView<'a, f64>;

pub type LocalCoord = Coord;
pub type LocalCoordRef<'a> = CoordRef<'a>;

pub type BaryCoord = Coord;
pub type BaryCoordRef<'a> = CoordRef<'a>;

pub type AmbientCoord = Coord;
pub type AmbientCoordRef<'a> = CoordRef<'a>;

pub fn standard_coord_complex(dim: Dim) -> (Complex, MeshVertexCoords) {
  let topology = Complex::standard(dim);

  let coords = topology
    .vertices()
    .iter()
    .map(|v| v.kidx())
    .map(|v| {
      let mut vec = na::DVector::zeros(dim);
      if v > 0 {
        vec[v - 1] = 1.0;
      }
      vec
    })
    .collect_vec();
  let coords = na::DMatrix::from_columns(&coords);
  let coords = MeshVertexCoords::new(coords);

  (topology, coords)
}

#[derive(Debug, Clone)]
pub struct MeshVertexCoords {
  coord_matrix: na::DMatrix<f64>,
}

impl MeshVertexCoords {
  pub fn standard(ndim: Dim) -> Self {
    SimplexCoords::standard(ndim).vertices
  }
  pub fn new(coord_matrix: na::DMatrix<f64>) -> Self {
    Self { coord_matrix }
  }

  pub fn matrix(&self) -> &na::DMatrix<f64> {
    &self.coord_matrix
  }
  pub fn matrix_mut(&mut self) -> &mut na::DMatrix<f64> {
    &mut self.coord_matrix
  }
  pub fn into_matrix(self) -> na::DMatrix<f64> {
    self.coord_matrix
  }

  fn swap_coords(&mut self, icol: usize, jcol: usize) {
    self.coord_matrix.swap_columns(icol, jcol)
  }
}

impl From<na::DMatrix<f64>> for MeshVertexCoords {
  fn from(matrix: na::DMatrix<f64>) -> Self {
    Self::new(matrix)
  }
}

impl From<&[Coord]> for MeshVertexCoords {
  fn from(vectors: &[Coord]) -> Self {
    let matrix = na::DMatrix::from_columns(vectors);
    Self::new(matrix)
  }
}

impl MeshVertexCoords {
  pub fn dim(&self) -> Dim {
    self.coord_matrix.nrows()
  }
  pub fn nvertices(&self) -> usize {
    self.coord_matrix.ncols()
  }

  pub fn coord(&self, ivertex: VertexIdx) -> CoordRef {
    self.coord_matrix.column(ivertex)
  }

  pub fn coord_iter(
    &self,
  ) -> na::iter::ColumnIter<'_, f64, na::Dyn, na::Dyn, na::VecStorage<f64, na::Dyn, na::Dyn>> {
    self.coord_matrix.column_iter()
  }

  pub fn coord_iter_mut(
    &mut self,
  ) -> na::iter::ColumnIterMut<'_, f64, na::Dyn, na::Dyn, na::VecStorage<f64, na::Dyn, na::Dyn>> {
    self.coord_matrix.column_iter_mut()
  }

  pub fn to_edge_lengths(&self, topology: &Complex) -> MeshEdgeLengths {
    let edges = topology.edges();
    let mut edge_lengths = na::DVector::zeros(edges.len());
    for (iedge, edge) in edges.iter().enumerate() {
      let [vi, vj] = edge.raw().clone().try_into().unwrap();
      let length = (self.coord(vj) - self.coord(vi)).norm();
      edge_lengths[iedge] = length;
    }
    // SAFETY: Edge Lengths come from a coordinate realizations.
    MeshEdgeLengths::new_unchecked(edge_lengths)
  }
}

impl MeshVertexCoords {
  pub fn embed_euclidean(mut self, dim: Dim) -> MeshVertexCoords {
    let old_dim = self.coord_matrix.nrows();
    self.coord_matrix = self.coord_matrix.insert_rows(old_dim, dim - old_dim, 0.0);
    self
  }
}
