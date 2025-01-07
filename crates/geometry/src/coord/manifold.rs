pub mod cartesian;
pub mod dim3;
pub mod gmsh;

use super::VertexCoords;
use crate::metric::manifold::MetricComplex;

use common::linalg::DMatrixExt;
use index_algebra::sign::Sign;
use topology::{complex::ManifoldComplex, simplex::Simplex, skeleton::ManifoldSkeleton, Dim};

#[derive(Debug, Clone)]
pub struct CoordSkeleton {
  skeleton: ManifoldSkeleton,
  coords: VertexCoords,
}
impl CoordSkeleton {
  pub fn new(skeleton: ManifoldSkeleton, coords: VertexCoords) -> Self {
    Self { skeleton, coords }
  }
  pub fn dim_embedded(&self) -> Dim {
    self.coords.dim()
  }
  pub fn dim_intrinsic(&self) -> Dim {
    self.skeleton.dim()
  }

  pub fn skeleton(&self) -> &ManifoldSkeleton {
    &self.skeleton
  }
  pub fn coords(&self) -> &VertexCoords {
    &self.coords
  }
  pub fn coords_mut(&mut self) -> &mut VertexCoords {
    &mut self.coords
  }
  pub fn into_parts(self) -> (ManifoldSkeleton, VertexCoords) {
    (self.skeleton, self.coords)
  }

  pub fn embed_euclidean(mut self, dim: Dim) -> CoordSkeleton {
    self.coords = self.coords.embed_euclidean(dim);
    self
  }

  pub fn into_metric_complex(self) -> (MetricComplex, VertexCoords) {
    let Self { skeleton, coords } = self;
    let complex = ManifoldComplex::from_facet_skeleton(skeleton);
    let edges = complex.edges();
    let edges = edges
      .iter()
      .map(|e| e.simplex_set().clone().try_into().unwrap());
    let edge_lengths = coords.to_edge_lengths(edges);
    (MetricComplex::new(complex, edge_lengths), coords)
  }
}

pub struct CoordComplex {
  _complex: ManifoldComplex,
  _coords: VertexCoords,
}

pub struct CoordSimplex {
  pub vertices: VertexCoords,
}
impl CoordSimplex {
  pub fn new(vertices: impl Into<VertexCoords>) -> Self {
    let vertices = vertices.into();
    Self { vertices }
  }

  pub fn standard(ndim: Dim) -> Self {
    let nvertices = ndim + 1;
    let mut vertices = na::DMatrix::<f64>::zeros(ndim, nvertices);
    for i in 0..ndim {
      vertices[(i, i + 1)] = 1.0;
    }
    Self::new(vertices)
  }

  pub fn from_simplex<O>(simp: &Simplex<O>, coords: &VertexCoords) -> CoordSimplex
  where
    O: index_algebra::variants::SetOrder,
  {
    let mut vert_coords = na::DMatrix::zeros(coords.dim(), simp.len());
    for (i, v) in simp.iter().enumerate() {
      vert_coords.set_column(i, &coords.coord(v));
    }
    CoordSimplex::new(vert_coords)
  }
}
impl CoordSimplex {
  pub fn nvertices(&self) -> usize {
    self.vertices.nvertices()
  }
  pub fn dim_intrinsic(&self) -> Dim {
    self.nvertices() - 1
  }
  pub fn dim_embedded(&self) -> Dim {
    self.vertices.dim()
  }
  pub fn is_euclidean(&self) -> bool {
    self.dim_intrinsic() == self.dim_embedded()
  }

  pub fn spanning_vectors(&self) -> na::DMatrix<f64> {
    let mut mat = na::DMatrix::zeros(self.dim_embedded(), self.dim_intrinsic());
    let v0 = self.vertices.coord(0);
    for (i, vi) in self.vertices.coord_iter().skip(1).enumerate() {
      let v0i = vi - &v0;
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
