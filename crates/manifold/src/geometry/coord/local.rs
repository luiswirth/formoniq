use super::{
  AmbientCoord, AmbientCoordRef, BaryCoord, BaryCoordRef, Coord, CoordRef, LocalCoord,
  LocalCoordRef, MeshVertexCoords,
};
use crate::{
  geometry::{metric::SimplexLengths, refsimp_vol},
  topology::{complex::handle::SimplexHandle, simplex::Simplex},
  Dim,
};

use common::{affine::AffineTransform, combo::Sign, gramian::Gramian};
use tracing::warn;

#[derive(Debug, Clone)]
pub struct SimplexCoords {
  pub vertices: MeshVertexCoords,
}

impl SimplexCoords {
  pub fn new(vertices: na::DMatrix<f64>) -> Self {
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
  pub fn from_simplex_and_coords(simp: &Simplex, coords: &MeshVertexCoords) -> SimplexCoords {
    let mut vert_coords = na::DMatrix::zeros(coords.dim(), simp.nvertices());
    for (i, v) in simp.iter().enumerate() {
      vert_coords.set_column(i, &coords.coord(v));
    }
    SimplexCoords::new(vert_coords)
  }

  pub fn nvertices(&self) -> usize {
    self.vertices.nvertices()
  }
  pub fn dim_intrinsic(&self) -> Dim {
    self.nvertices() - 1
  }
  pub fn dim_ambient(&self) -> Dim {
    self.vertices.dim()
  }
  pub fn is_same_dim(&self) -> bool {
    self.dim_intrinsic() == self.dim_ambient()
  }

  pub fn coord(&self, ivertex: usize) -> CoordRef {
    self.vertices.coord(ivertex)
  }

  pub fn base_vertex(&self) -> CoordRef {
    self.coord(0)
  }
  pub fn spanning_vector(&self, i: usize) -> na::DVector<f64> {
    assert!(i < self.dim_intrinsic());
    self.coord(i + 1) - self.base_vertex()
  }
  pub fn spanning_vectors(&self) -> na::DMatrix<f64> {
    let mut mat = na::DMatrix::zeros(self.dim_ambient(), self.dim_intrinsic());
    let v0 = self.base_vertex();
    for (i, vi) in self.vertices.coord_iter().skip(1).enumerate() {
      let v0i = vi - v0;
      mat.set_column(i, &v0i);
    }
    mat
  }
  pub fn metric_tensor(&self) -> Gramian {
    Gramian::from_euclidean_vectors(self.spanning_vectors())
  }

  pub fn det(&self) -> f64 {
    let det = if self.is_same_dim() {
      self.spanning_vectors().determinant()
    } else {
      Gramian::from_euclidean_vectors(self.spanning_vectors()).det_sqrt()
    };
    refsimp_vol(self.dim_intrinsic()) * det
  }
  pub fn vol(&self) -> f64 {
    self.det().abs()
  }
  pub fn orientation(&self) -> Sign {
    Sign::from_f64(self.det()).unwrap()
  }

  pub fn linear_transform(&self) -> na::DMatrix<f64> {
    self.spanning_vectors()
  }
  pub fn affine_transform(&self) -> AffineTransform {
    let translation = self.base_vertex().into_owned();
    let linear = self.linear_transform();
    AffineTransform::new(translation, linear)
  }
  pub fn local2global<'a>(&self, local: impl Into<LocalCoordRef<'a>>) -> AmbientCoord {
    let local = local.into();
    self.affine_transform().apply_forward(local)
  }
  pub fn global2local<'a>(&self, global: impl Into<AmbientCoordRef<'a>>) -> LocalCoord {
    let global = global.into();
    self.affine_transform().apply_backward(global)
  }
  pub fn global2bary<'a>(&self, global: impl Into<AmbientCoordRef<'a>>) -> BaryCoord {
    let local = self.global2local(global);
    let bary0 = 1.0 - local.sum();
    local.insert_row(0, bary0)
  }

  pub fn bary2global<'a>(&self, bary: impl Into<BaryCoordRef<'a>>) -> AmbientCoord {
    let bary = bary.into();
    self
      .vertices
      .coord_iter()
      .zip(bary.iter())
      .map(|(vi, &baryi)| baryi * vi)
      .sum()
  }

  /// Exterior derivative / total differential of barycentric coordinate functions
  /// in the rows(!) of a matrix.
  pub fn difbarys(&self) -> na::DMatrix<f64> {
    let difs = self.linear_transform().pseudo_inverse(1e-12).unwrap();
    let mut grads = difs.insert_row(0, 0.0);
    grads.set_row(0, &-grads.row_sum());
    grads
  }

  pub fn barycenter(&self) -> Coord {
    let mut barycenter = na::DVector::zeros(self.dim_ambient());
    self.vertices.coord_iter().for_each(|v| barycenter += v);
    barycenter /= self.nvertices() as f64;
    barycenter
  }
  pub fn is_coord_inside(&self, global: CoordRef) -> bool {
    let bary = self.global2bary(global);
    is_bary_inside(&bary)
  }

  /// Here we mean subsequence simplicies.
  pub fn subsimps(&self, sub_dim: Dim) -> impl Iterator<Item = SimplexCoords> + use<'_> {
    Simplex::standard(self.dim_intrinsic())
      .subsequences(sub_dim)
      .map(|edge| Self::from_simplex_and_coords(&edge, &self.vertices))
  }
  pub fn edges(&self) -> impl Iterator<Item = SimplexCoords> + use<'_> {
    self.subsimps(1)
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
  pub fn flipped_orientation(mut self) -> Self {
    self.flip_orientation();
    self
  }

  pub fn to_lengths(&self) -> SimplexLengths {
    SimplexLengths::from_coords(self)
  }
}

pub fn local_to_bary_coords<'a>(local: impl Into<CoordRef<'a>>) -> Coord {
  let local = local.into();
  let bary0 = 1.0 - local.sum();
  local.insert_row(0, bary0)
}

pub fn is_bary_inside<'a>(bary: impl Into<CoordRef<'a>>) -> bool {
  bary.into().iter().all(|&b| (0.0..=1.0).contains(&b))
}
pub fn ref_barycenter(dim: Dim) -> Coord {
  let nvertices = dim + 1;
  let value = 1.0 / nvertices as f64;
  na::DVector::from_element(dim, value)
}
pub fn ref_bary<'a>(ivertex: usize, coord: impl Into<CoordRef<'a>>) -> f64 {
  let coord = coord.into();
  let dim = coord.len();
  assert!(ivertex <= dim);
  if ivertex == 0 {
    1.0 - coord.sum()
  } else {
    coord[ivertex - 1]
  }
}
pub fn ref_gradbary(dim: Dim, ivertex: usize) -> na::DVector<f64> {
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
impl SimplexHandleExt for SimplexHandle<'_> {
  fn coord_simplex(&self, coords: &MeshVertexCoords) -> SimplexCoords {
    SimplexCoords::from_simplex_and_coords(self.raw(), coords)
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
        let computed = simp.global2bary(pos);
        for ibary in 0..simp.nvertices() {
          let expected = ref_bary(ibary, pos);
          assert_eq!(computed[ibary], expected);
        }
      }
    }
  }

  #[test]
  fn standard_gradbarys() {
    for dim in 0..=4 {
      let simp = SimplexCoords::standard(dim);
      let computed = simp.difbarys();
      for ibary in 0..simp.nvertices() {
        let expected = ref_gradbary(dim, ibary);
        assert_eq!(computed.column(ibary), expected);
      }
    }
  }
}
