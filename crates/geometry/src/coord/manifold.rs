pub mod cartesian;
pub mod dim3;
pub mod gmsh;
pub mod vtk;

use super::{Coord, CoordRef, MeshVertexCoords};
use crate::metric::ref_vol;

use common::linalg::DMatrixExt;
use index_algebra::sign::Sign;
use itertools::Itertools;
use topology::{
  complex::{dim::DimInfoProvider, handle::SimplexHandle, TopologyComplex},
  simplex::Simplex,
  Dim,
};
use tracing::warn;

pub fn standard_coord_complex(dim: Dim) -> (TopologyComplex, MeshVertexCoords) {
  let topology = TopologyComplex::standard(dim);

  let coords = topology
    .vertices()
    .handle_iter()
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
pub struct SimplexCoords {
  pub vertices: MeshVertexCoords,
}
impl SimplexCoords {
  pub fn new(vertices: impl Into<MeshVertexCoords>) -> Self {
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

  pub fn from_simplex_and_coords<O>(simp: &Simplex<O>, coords: &MeshVertexCoords) -> SimplexCoords
  where
    O: index_algebra::variants::SetOrder,
  {
    let mut vert_coords = na::DMatrix::zeros(coords.dim(), simp.nvertices());
    for (i, v) in simp.vertices.iter().enumerate() {
      vert_coords.set_column(i, &coords.coord(v));
    }
    SimplexCoords::new(vert_coords)
  }

  pub fn edges(&self) -> impl Iterator<Item = SimplexCoords> + use<'_> {
    Simplex::standard(self.nvertices())
      .subsimps(1)
      .map(|edge| Self::from_simplex_and_coords(&edge, &self.vertices))
  }
}
impl SimplexCoords {
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

  pub fn base_vertex(&self) -> Coord {
    self.vertices.coord(0)
  }
  pub fn spanning_vector(&self, i: usize) -> na::DVector<f64> {
    assert!(i < self.dim_intrinsic());
    self.vertices.coord(i + 1) - self.base_vertex()
  }
  pub fn spanning_vectors(&self) -> na::DMatrix<f64> {
    let mut mat = na::DMatrix::zeros(self.dim_embedded(), self.dim_intrinsic());
    let v0 = self.base_vertex();
    for (i, vi) in self.vertices.coord_iter().skip(1).enumerate() {
      let v0i = vi - &v0;
      mat.set_column(i, &v0i);
    }
    mat
  }

  pub fn det(&self) -> f64 {
    let det = if self.is_euclidean() {
      self.spanning_vectors().determinant()
    } else {
      self.spanning_vectors().gram_det_sqrt()
    };
    ref_vol(self.dim_intrinsic()) * det
  }
  pub fn vol(&self) -> f64 {
    self.det().abs()
  }
  pub fn orientation(&self) -> Sign {
    Sign::from_f64(self.det())
  }

  pub fn swap_vertices(&mut self, icol: usize, jcol: usize) {
    self.vertices.swap_coords(icol, jcol)
  }
  pub fn flip_orientation(&mut self) {
    if self.nvertices() >= 2 {
      self.swap_vertices(0, 1)
    } else {
      warn!("Cannot flip CoordSimplex with less than 2 vertices.")
    }
  }

  pub fn linear_transform(&self) -> na::DMatrix<f64> {
    self.spanning_vectors()
  }

  pub fn affine_diffeomorphism(&self) -> AffineDiffeomorphism {
    let translation = self.base_vertex();
    let linear = self.linear_transform();
    AffineDiffeomorphism::from_forward(translation, linear)
  }

  pub fn local_to_global_coord<'a>(&self, local: impl Into<CoordRef<'a>>) -> Coord {
    let local = local.into();
    self.linear_transform() * local + self.base_vertex()
  }
  pub fn global_to_local_coord(&self, global: CoordRef) -> Coord {
    let linear_transform = self.linear_transform();
    if linear_transform.is_empty() {
      return Coord::default();
    }

    linear_transform
      .svd(true, true)
      .solve(&(global - self.base_vertex()), 1e-12)
      .unwrap()
  }

  pub fn global_to_bary_coord<'a>(&self, global: impl Into<CoordRef<'a>>) -> Coord {
    let global = global.into();
    local_to_bary_coord(&self.global_to_local_coord(global))
  }

  pub fn gradbary(&self, i: usize) -> na::DVector<f64> {
    if i == 0 {
      let spanning = self.spanning_vectors();
      -spanning.column_sum()
    } else {
      self.spanning_vector(i - 1)
    }
  }

  pub fn gradbarys(&self) -> na::DMatrix<f64> {
    let spanning = self.spanning_vectors();
    let difbary0 = -spanning.column_sum();
    let mut difbarys = spanning.insert_column(0, 0.0);
    difbarys.set_column(0, &difbary0);
    difbarys
  }

  pub fn barycenter(&self) -> Coord {
    let mut barycenter = na::DVector::zeros(self.dim_embedded());
    self.vertices.coord_iter().for_each(|v| barycenter += v);
    barycenter /= self.nvertices() as f64;
    barycenter
  }
  pub fn is_coord_inside(&self, global: CoordRef) -> bool {
    let bary = self.global_to_bary_coord(global);
    bary.iter().all(|&b| (0.0..=1.0).contains(&b))
  }
}

pub fn local_to_bary_coord<'a>(local: impl Into<CoordRef<'a>>) -> Coord {
  let local = local.into();
  let bary0 = 1.0 - local.sum();
  local.insert_row(0, bary0)
}

pub fn standard_bary<'a>(ivertex: usize, coord: impl Into<CoordRef<'a>>) -> f64 {
  let coord = coord.into();
  let dim = coord.len();
  assert!(ivertex <= dim);
  if ivertex == 0 {
    1.0 - coord.sum()
  } else {
    coord[ivertex - 1]
  }
}

pub fn standard_gradbary(dim: Dim, ivertex: usize) -> na::DVector<f64> {
  assert!(ivertex <= dim);
  if ivertex == 0 {
    na::DVector::from_element(dim, -1.0)
  } else {
    let mut v = na::DVector::zeros(dim);
    v[ivertex - 1] = 1.0;
    v
  }
}

pub trait SimplexHandleExt {
  fn coord_simplex(&self, coords: &MeshVertexCoords) -> SimplexCoords;
}
impl<'c, D: DimInfoProvider> SimplexHandleExt for SimplexHandle<'c, D> {
  fn coord_simplex(&self, coords: &MeshVertexCoords) -> SimplexCoords {
    SimplexCoords::from_simplex_and_coords(self.simplex_set(), coords)
  }
}

pub struct AffineDiffeomorphism {
  translation: na::DVector<f64>,
  linear: na::DMatrix<f64>,
  linear_svd: Option<na::SVD<f64, na::Dyn, na::Dyn>>,
  linear_inv: na::DMatrix<f64>,
}
impl AffineDiffeomorphism {
  pub fn from_forward(translation: na::DVector<f64>, linear: na::DMatrix<f64>) -> Self {
    let linear_svd = linear.is_empty().then(|| linear.clone().svd(true, true));
    let linear_inv = if linear.is_empty() {
      na::DMatrix::zeros(0, 0)
    } else if linear.is_square() {
      linear.clone().try_inverse().unwrap()
    } else {
      linear.clone().pseudo_inverse(1e-12).unwrap()
    };
    Self {
      translation,
      linear,
      linear_svd,
      linear_inv,
    }
  }

  pub fn apply_forward(&self, coord: CoordRef) -> Coord {
    &self.linear * coord + &self.translation
  }
  pub fn apply_backward(&self, coord: CoordRef) -> Coord {
    let Some(svd) = &self.linear_svd else {
      return Coord::default();
    };
    svd.solve(&(coord - &self.translation), 1e-12).unwrap()
  }
  pub fn linear_inv(&self) -> &na::DMatrix<f64> {
    &self.linear_inv
  }
}

#[cfg(test)]
mod test {
  use super::*;

  #[test]
  fn standard_barys() {
    for dim in 0..=4 {
      let simp = SimplexCoords::standard(dim);
      for pos in simp.vertices.coord_iter() {
        let computed = simp.global_to_bary_coord(pos);
        for ibary in 0..simp.nvertices() {
          let expected = standard_bary(ibary, pos);
          assert_eq!(computed[ibary], expected);
        }
      }
    }
  }

  #[test]
  fn standard_gradbarys() {
    for dim in 0..=4 {
      let simp = SimplexCoords::standard(dim);
      let computed = simp.gradbarys();
      for ibary in 0..simp.nvertices() {
        let expected = standard_gradbary(dim, ibary);
        assert_eq!(computed.column(ibary), expected);
      }
    }
  }
}
