use common::linalg::DMatrixExt;
use geometry::coord::VertexCoords;
use index_algebra::sign::Sign;

use crate::{
  complex::Complex,
  simplicial::{OrderedVertplex, OrientedVertplex, SimplexExt, SimplicialTopology},
  Dim, RiemannianComplex,
};

#[derive(Debug, Clone)]
pub struct CoordManifold {
  facets: SimplicialTopology,
  coords: VertexCoords,
}
impl CoordManifold {
  pub fn new(facets: Vec<OrientedVertplex>, coords: VertexCoords) -> Self {
    for facet in &facets {
      let coord_cell = CoordSimplex::from_vertplex(&facet.clone().forget_sign(), &coords);
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

  pub fn into_riemannian_complex(self) -> (RiemannianComplex, VertexCoords) {
    let Self { facets, coords } = self;
    let complex = Complex::new(facets.clone());
    let edges = complex
      .edges()
      .keys()
      .map(|e| e.indices().try_into().unwrap());
    let edge_lenghts = coords.to_edge_lengths(edges);
    (
      RiemannianComplex::new(facets, complex, edge_lenghts),
      coords,
    )
  }
}

pub struct CoordSimplex {
  pub vertices: na::DMatrix<f64>,
}
impl CoordSimplex {
  pub fn new(vertices: na::DMatrix<f64>) -> Self {
    Self { vertices }
  }

  pub fn standard(ndim: Dim) -> Self {
    let nvertices = ndim + 1;
    let mut vertices = na::DMatrix::<f64>::zeros(ndim, nvertices);
    for i in 0..ndim {
      vertices[(i, i + 1)] = 1.0;
    }
    Self { vertices }
  }

  pub fn from_vertplex(simp: &OrderedVertplex, coords: &VertexCoords) -> CoordSimplex {
    let mut vert_coords = na::DMatrix::zeros(coords.dim(), simp.len());
    for (i, &v) in simp.iter().enumerate() {
      vert_coords.set_column(i, &coords.coord(v));
    }
    CoordSimplex::new(vert_coords)
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

  pub fn integrate_diff_form<F>(&self, _f: F) -> f64
  where
    F: Fn(na::DVector<f64>) -> na::DVector<f64>,
  {
    todo!("trapezoidal rule")
  }
}
