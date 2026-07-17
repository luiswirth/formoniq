//! The affine parametrization of a cell: the chart, realized in coordinates.
//!
//! A [`SimplexCoords`] is the vertex coordinates of a simplex, and the map it
//! carries is $psi_K: hat(K) -> RR^N$, $x |-> v_0 + A x$ -- from the *chart* of
//! the cell out into the ambient space. It is a parametrization, not a chart:
//! the chart runs the other way and is the barycentric one, which exists on
//! every geometry ([`crate::atlas`]).
//!
//! Everything here therefore presupposes an embedding, and is extrinsic. The
//! metric it induces (through [`metric_tensor`](SimplexCoords::metric_tensor))
//! and the edge lengths it realizes (through
//! [`to_lengths`](SimplexCoords::to_lengths)) are the two bridges down into the
//! intrinsic layer, and they run downward only: the metric layer never learns
//! that coordinates exist.

use super::{
  mesh::MeshCoords, CoTangentVector, CoTangentVectorRef, Coord, CoordRef, TangentVector,
  TangentVectorRef,
};
use crate::{
  atlas::{is_bary_inside, local2bary, refsimp_vol, Bary, BaryRef, Local, LocalRef},
  geometry::metric::simplex::SimplexLengths,
  topology::{handle::SimplexRef, simplex::Simplex},
  Dim,
};

use common::{
  affine::AffineTransform,
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
  /// The standard simplex: the coordinate realization of the reference cell,
  /// whose ambient coordinates *are* the local coordinates of the chart.
  pub fn standard(ndim: Dim) -> Self {
    Self::new(crate::atlas::ref_vertices(ndim))
  }
  pub fn from_simplex_and_coords(simp: &Simplex, coords: &MeshCoords) -> SimplexCoords {
    let mut vert_coords = Matrix::zeros(coords.dim(), simp.nvertices());
    for (i, v) in simp.iter().enumerate() {
      vert_coords.set_column(i, &coords.coord(v).view());
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

  pub fn coord(&self, ivertex: usize) -> CoordRef<'_> {
    self.vertices.coord(ivertex)
  }
  pub fn coord_iter(&self) -> impl ExactSizeIterator<Item = CoordRef<'_>> {
    self.vertices.coord_iter()
  }

  pub fn base_vertex(&self) -> CoordRef<'_> {
    self.coord(0)
  }

  pub fn spanning_vector(&self, i: usize) -> TangentVector {
    assert!(i < self.dim_intrinsic());
    self.coord(i + 1) - self.base_vertex()
  }
  /// The spanning vectors $A$ of the parametrization, as the columns of an
  /// ambient-by-intrinsic matrix: its linear part.
  pub fn spanning_vectors(&self) -> Matrix {
    let mut mat = Matrix::zeros(self.dim_ambient(), self.dim_intrinsic());
    for i in 0..self.dim_intrinsic() {
      mat.set_column(i, &self.spanning_vector(i));
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

  /// The linear part $A$ of the parametrization: the differential
  /// $dif psi_K$, constant because the parametrization is affine.
  pub fn linear_transform(&self) -> Matrix {
    self.spanning_vectors()
  }
  /// The pseudo-inverse $A^+$ of the linear part.
  ///
  /// A genuine inverse only when the cell is of full ambient dimension. On an
  /// embedded manifold it is the Moore-Penrose one, which annihilates the normal
  /// space -- a metric-dependent choice, and hence not canonical.
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
    let translation = self.base_vertex().view().into_owned();
    let linear = self.linear_transform();
    AffineTransform::new(translation, linear)
  }

  /// $psi_K$: the parametrization, from the chart out into the ambient space.
  pub fn local2global<'a>(&self, local: impl Into<LocalRef<'a>>) -> Coord {
    Coord::new(self.affine_transform().apply_forward(local.into().view()))
  }
  /// $psi_K^(-1)$: back from the ambient space into the chart.
  pub fn global2local<'a>(&self, global: impl Into<CoordRef<'a>>) -> Local {
    Local::new(self.affine_transform().apply_backward(global.into().view()))
  }
  pub fn global2bary<'a>(&self, global: impl Into<CoordRef<'a>>) -> Bary {
    local2bary(&self.global2local(global))
  }
  pub fn bary2global<'a>(&self, bary: impl Into<BaryRef<'a>>) -> Coord {
    let bary = bary.into();
    let global: Vector = self
      .coord_iter()
      .zip(bary.view().iter())
      .map(|(vi, &baryi)| baryi * vi.view())
      .sum();
    Coord::new(global)
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
    self.coord_iter().for_each(|v| barycenter += v.view());
    barycenter /= self.nvertices() as f64;
    Coord::new(barycenter)
  }
  pub fn is_global_inside(&self, global: CoordRef) -> bool {
    is_bary_inside(&self.global2bary(global))
  }
}

impl SimplexCoords {
  /// Coordinate subsimplices.
  pub fn subsimps(&self, sub_dim: Dim) -> impl Iterator<Item = SimplexCoords> + use<'_> {
    let simp = Simplex::standard(self.dim_intrinsic());
    simp
      .subsimps(sub_dim)
      .collect::<Vec<_>>()
      .into_iter()
      .map(|sub| Self::from_simplex_and_coords(&sub, &self.vertices))
  }
  pub fn edges(&self) -> impl Iterator<Item = SimplexCoords> + use<'_> {
    self.subsimps(1)
  }

  pub fn swap_vertices(&mut self, icol: usize, jcol: usize) {
    self.vertices.swap_coords(icol, jcol);
  }
  pub fn flip_orientation(&mut self) {
    if self.nvertices() >= 2 {
      self.swap_vertices(0, 1);
    } else {
      warn!("Cannot flip CoordSimplex with less than 2 vertices.");
    }
  }
  pub fn flipped_orientation(mut self) -> Self {
    self.flip_orientation();
    self
  }

  /// The Regge edge lengths this coordinate realization has: the bridge from
  /// the extrinsic layer down into the intrinsic one.
  pub fn to_lengths(&self) -> SimplexLengths {
    let lengths: Vec<f64> = self.edges().map(|e| e.vol()).collect();
    // SAFETY: Edge lengths stem from a realization already.
    SimplexLengths::new_unchecked(lengths.into(), self.dim_intrinsic())
  }
}

/// The affine parametrization of a cell, given an embedding: an `exterior`-free
/// coordinate construction on a topology handle, which is how invariant 1 is
/// upheld below crate granularity.
pub trait SimplexRefExt {
  fn coord_simplex(&self, coords: &MeshCoords) -> SimplexCoords;
}
impl SimplexRefExt for SimplexRef<'_> {
  fn coord_simplex(&self, coords: &MeshCoords) -> SimplexCoords {
    SimplexCoords::from_simplex_and_coords(self.simplex(), coords)
  }
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::{
    atlas::{ref_bary, ref_difbarys},
    geometry::metric::simplex::SimplexLengths,
  };

  use approx::assert_relative_eq;

  /// The standard simplex is the coordinate realization of the reference chart:
  /// its ambient coordinates *are* the local ones.
  #[test]
  fn standard_barys() {
    for dim in 0..=4 {
      let simp = SimplexCoords::standard(dim);
      for pos in simp.coord_iter() {
        let local = Local::new(pos.view().into_owned());
        let computed = simp.global2bary(pos);
        for ibary in 0..simp.nvertices() {
          let expected = ref_bary(ibary, &local);
          assert_eq!(computed[ibary], expected);
        }
      }
    }
  }

  /// The barycentric differentials of the standard simplex are the metric-free
  /// reference ones -- which is what lets the intrinsic Whitney forms use
  /// [`ref_difbarys`] and never touch coordinates.
  #[test]
  fn standard_difbarys() {
    for dim in 0..=4 {
      let computed = SimplexCoords::standard(dim).difbarys();
      assert_relative_eq!(computed, ref_difbarys(dim), epsilon = 1e-12);
    }
  }

  /// The standard coordinate simplex realizes the standard edge lengths: the
  /// two descriptions of the reference cell agree, extrinsic and intrinsic.
  #[test]
  fn ref_coords_realize_ref_lengths() {
    for dim in 0..=4 {
      let coords = SimplexCoords::standard(dim);
      let lengths = coords.to_lengths();
      assert_relative_eq!(lengths.vector(), SimplexLengths::standard(dim).vector());
      assert_relative_eq!(coords.vol(), lengths.vol());
    }
  }

  /// The parametrization and its inverse are mutually inverse on the chart.
  #[test]
  fn local_global_roundtrip() {
    for dim in 1..=3 {
      let simp = SimplexCoords::standard(dim);
      let local = Local::from_iterator(dim, (0..dim).map(|i| 0.1 * (i + 1) as f64));
      let global = simp.local2global(&local);
      assert_relative_eq!(
        simp.global2local(&global).vector(),
        local.vector(),
        epsilon = 1e-12
      );
    }
  }
}
