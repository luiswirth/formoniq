use super::{
  complex::Complex,
  geometry::EdgeLengths,
  simplicial::{OrderedVertplex, OrientedVertplex, SimplexExt},
  RiemannianComplex,
};
use crate::{combo::Sign, linalg::DMatrixExt as _, mesh::VertexIdx, Dim};

#[derive(Debug, Clone)]
pub struct CoordManifold {
  facets: Vec<OrientedVertplex>,
  coords: VertexCoords,
}
impl CoordManifold {
  pub fn new(facets: Vec<OrientedVertplex>, coords: VertexCoords) -> Self {
    for facet in &facets {
      let coord_cell = coords.coord_simplex(&facet.clone().forget_sign());
      assert!(
        coord_cell.orientation() * facet.sign() == Sign::Pos,
        "Facets must be positively oriented."
      );
    }

    Self { facets, coords }
  }
  pub fn dim_embedded(&self) -> Dim {
    self.coords.dim()
  }
  pub fn dim_intrinsic(&self) -> Dim {
    self.facets[0].dim()
  }

  pub fn facets(&self) -> &[OrientedVertplex] {
    &self.facets
  }
  pub fn coords(&self) -> &VertexCoords {
    &self.coords
  }
  pub fn coords_mut(&mut self) -> &mut VertexCoords {
    &mut self.coords
  }
  pub fn into_parts(self) -> (Vec<OrientedVertplex>, VertexCoords) {
    (self.facets, self.coords)
  }

  pub fn embed_euclidean(mut self, dim: Dim) -> CoordManifold {
    self.coords = self.coords.embed_euclidean(dim);
    self
  }

  // TODO: consume vs clone -> make perfect split
  pub fn to_riemannian_complex(&self) -> RiemannianComplex {
    let Self { facets, coords } = self;
    let complex = Complex::new(facets.clone());
    let edge_lenghts = coords.to_edge_lengths(&complex);
    RiemannianComplex::new(facets.clone(), complex, edge_lenghts)
  }
}

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

  pub fn to_edge_lengths(&self, complex: &Complex) -> EdgeLengths {
    let mut edge_lengths = na::DVector::zeros(complex.edges().len());
    for (iedge, edge) in complex.edges().keys().enumerate() {
      let &[vi, vj] = edge.indices() else {
        unreachable!()
      };
      let length = (self.coord(vj) - self.coord(vi)).norm();
      edge_lengths[iedge] = length;
    }
    EdgeLengths::new(edge_lengths)
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
