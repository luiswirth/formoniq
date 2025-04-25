use super::{
  mesh::MeshCoords, CoTangentVector, CoTangentVectorRef, Coord, CoordRef, TangentVector,
  TangentVectorRef,
};
use crate::{
  geometry::{metric::simplex::SimplexLengths, refsimp_vol},
  topology::{handle::SimplexHandle, simplex::Simplex},
  Dim,
};

use approx::assert_relative_eq;
use common::{
  affine::AffineTransform,
  combo::Sign,
  gramian::Gramian,
  linalg::nalgebra::{Matrix, Vector},
};
use tracing::warn;

#[derive(Debug, Clone)]
pub struct SimplexCoords {
  pub vertices: MeshCoords,
}

impl SimplexCoords {
  pub fn new(vertices: Matrix) -> Self {
    let vertices = vertices.into();
    Self { vertices }
  }
  pub fn standard(ndim: Dim) -> Self {
    let nvertices = ndim + 1;
    let mut vertices = Matrix::<f64>::zeros(ndim, nvertices);
    for i in 0..ndim {
      vertices[(i, i + 1)] = 1.0;
    }
    Self::new(vertices)
  }
  pub fn from_simplex_and_coords(simp: &Simplex, coords: &MeshCoords) -> SimplexCoords {
    let mut vert_coords = Matrix::zeros(coords.dim(), simp.nvertices());
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

  pub fn spanning_vector(&self, i: usize) -> TangentVector {
    assert!(i < self.dim_intrinsic());
    self.coord(i + 1) - self.base_vertex()
  }
  pub fn spanning_vectors(&self) -> Matrix {
    let mut mat = Matrix::zeros(self.dim_ambient(), self.dim_intrinsic());
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
      self.metric_tensor().det_sqrt()
    };
    refsimp_vol(self.dim_intrinsic()) * det
  }
  pub fn vol(&self) -> f64 {
    self.det().abs()
  }
  pub fn is_degenerate(&self) -> bool {
    self.vol() <= 1e-12
  }

  // TODO: makes only sense for coinciding ambient and intrinsic dim
  pub fn orientation(&self) -> Sign {
    Sign::from_f64(self.det()).unwrap()
  }

  pub fn linear_transform(&self) -> Matrix {
    self.spanning_vectors()
  }
  pub fn inv_linear_transform(&self) -> Matrix {
    if self.dim_intrinsic() == 0 {
      Matrix::zeros(0, 0)
    } else {
      self.linear_transform().pseudo_inverse(1e-12).unwrap()
    }
  }

  /// Local2Global Tangentvector
  pub fn pushforward_vector<'a>(&self, local: impl Into<TangentVectorRef<'a>>) -> TangentVector {
    self.linear_transform() * local.into()
  }
  /// Global2Local Cotangentvector
  pub fn pullback_covector<'a>(
    &self,
    global: impl Into<CoTangentVectorRef<'a>>,
  ) -> CoTangentVector {
    global.into() * self.linear_transform()
  }

  pub fn affine_transform(&self) -> AffineTransform {
    let translation = self.base_vertex().into_owned();
    let linear = self.linear_transform();
    AffineTransform::new(translation, linear)
  }
  pub fn local2global<'a>(&self, local: impl Into<CoordRef<'a>>) -> Coord {
    self.affine_transform().apply_forward(local.into())
  }
  pub fn global2local<'a>(&self, global: impl Into<CoordRef<'a>>) -> Coord {
    self.affine_transform().apply_backward(global.into())
  }
  pub fn global2bary<'a>(&self, global: impl Into<CoordRef<'a>>) -> Coord {
    local2bary(&self.global2local(global))
  }

  pub fn bary2global<'a>(&self, bary: impl Into<CoordRef<'a>>) -> Coord {
    self
      .vertices
      .coord_iter()
      .zip(bary.into().iter())
      .map(|(vi, &baryi)| baryi * vi)
      .sum()
  }

  /// Total differential of barycentric coordinate functions in the rows(!) of
  /// a matrix.
  pub fn difbarys(&self) -> Matrix {
    let difs = self.inv_linear_transform();
    let mut difs = difs.insert_row(0, 0.0);
    difs.set_row(0, &-difs.row_sum());
    difs
  }

  pub fn barycenter(&self) -> Coord {
    let mut barycenter = Vector::zeros(self.dim_ambient());
    self.vertices.coord_iter().for_each(|v| barycenter += v);
    barycenter /= self.nvertices() as f64;
    barycenter
  }
  pub fn is_global_inside(&self, global: CoordRef) -> bool {
    is_bary_inside(&self.global2bary(global))
  }
}
pub fn bary2local<'a>(bary: impl Into<CoordRef<'a>>) -> Coord {
  let bary = bary.into();
  bary.view_range(1.., ..).into()
}
pub fn local2bary<'a>(local: impl Into<CoordRef<'a>>) -> Coord {
  let local = local.into();
  let bary0 = 1.0 - local.sum();
  local.insert_row(0, bary0)
}

pub fn is_bary_inside<'a>(bary: impl Into<CoordRef<'a>>) -> bool {
  let bary = bary.into();
  assert_relative_eq!(bary.sum(), 1.0);
  bary.iter().all(|&b| (0.0..=1.0).contains(&b))
}

impl SimplexCoords {
  /// Coordinate subsequence simplicies.
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

pub fn ref_barycenter(dim: Dim) -> Coord {
  let nvertices = dim + 1;
  let value = 1.0 / nvertices as f64;
  Vector::from_element(dim, value)
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
pub fn ref_difbary(dim: Dim, ivertex: usize) -> CoTangentVector {
  assert!(ivertex <= dim);
  if ivertex == 0 {
    CoTangentVector::from_element(dim, -1.0)
  } else {
    let mut v = CoTangentVector::zeros(dim);
    v[ivertex - 1] = 1.0;
    v
  }
}

pub trait SimplexHandleExt {
  fn coord_simplex(&self, coords: &MeshCoords) -> SimplexCoords;
}
impl SimplexHandleExt for SimplexHandle<'_> {
  fn coord_simplex(&self, coords: &MeshCoords) -> SimplexCoords {
    SimplexCoords::from_simplex_and_coords(self, coords)
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
  fn standard_difbarys() {
    for dim in 0..=4 {
      let simp = SimplexCoords::standard(dim);
      let computed = simp.difbarys();
      for ibary in 0..simp.nvertices() {
        let expected = ref_difbary(dim, ibary);
        assert_eq!(computed.column(ibary), expected);
      }
    }
  }
}
