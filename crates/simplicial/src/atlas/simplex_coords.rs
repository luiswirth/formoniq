//! The affine realization of a simplex in a coordinate space.
//!
//! A [`SimplexCoords<S>`] is the vertex coordinates of a simplex in the space
//! `S`, and the map it carries is the affine parametrization $x |-> v_0 + A x$
//! from the *reference chart* $hat(K)$ of the cell into `S`. It is a
//! parametrization, not a chart: the chart runs the other way and is the
//! barycentric one, which exists on every geometry ([`crate::atlas`]).
//!
//! The space is a type parameter, and that is the whole point. The same
//! construction is used in two genuinely different spaces:
//!
//! - `SimplexCoords<Ambient>` is a cell embedded in $RR^N$ -- the extrinsic
//!   realization, whose induced metric and edge lengths (`geometry::coord`) are
//!   the bridges down into the intrinsic layer.
//! - `SimplexCoords<LocalCartesian>` is a simplex realized in a chart's own
//!   cartesian frame $RR^n$. [`standard`](SimplexCoords::standard) is the
//!   reference cell itself ("its ambient coordinates *are* its local
//!   coordinates"), and a sub-simplex of a refinement is the child realized in
//!   its parent's frame -- the map its metric is pulled back along.
//!
//! Everything here is affine and metric-free: it needs coordinates in *some*
//! space, never an inner product on that space. The metric a realization
//! induces is a `geometry::coord` concern layered on the `Ambient`
//! instantiation, and it is the only part that presupposes an embedding.

use super::{
  Bary, BaryRef, Local, LocalCartesian, LocalRef, is_bary_inside, local2bary, ref_vertices,
  refsimp_vol,
};
use crate::Dim;
use crate::linalg::{Matrix, RowVector, RowVectorView, Vector, VectorView};
use crate::topology::simplex::Simplex;

use coorder::{Ambient, CoordSpace, Coords, CoordsRef, affine::AffineTransform};
use tracing::warn;

use std::marker::PhantomData;

/// The vertex coordinates of a simplex realized in the coordinate space `S`, as
/// the columns of a matrix. The default space is [`Ambient`], the embedded case.
pub struct SimplexCoords<S: CoordSpace = Ambient> {
  vertices: Matrix,
  space: PhantomData<S>,
}

impl<S: CoordSpace> SimplexCoords<S> {
  pub fn new(vertices: Matrix) -> Self {
    Self {
      vertices,
      space: PhantomData,
    }
  }

  pub fn vertices(&self) -> &Matrix {
    &self.vertices
  }

  pub fn nvertices(&self) -> usize {
    self.vertices.ncols()
  }
  pub fn dim_intrinsic(&self) -> Dim {
    (self.nvertices() - 1).into()
  }
  /// The dimension of the coordinate space `S` the simplex is realized in.
  pub fn dim_space(&self) -> Dim {
    self.vertices.nrows().into()
  }
  /// Whether the realization is full-dimensional: the simplex spans its space,
  /// so the parametrization is square and invertible. Always true for
  /// [`LocalCartesian`]; false for a lower-dimensional cell embedded in a
  /// higher-dimensional `Ambient` space (a surface in $RR^3$).
  pub fn is_same_dim(&self) -> bool {
    self.dim_intrinsic() == self.dim_space()
  }

  pub fn coord(&self, ivertex: usize) -> CoordsRef<'_, S> {
    CoordsRef::new(self.vertices.column(ivertex))
  }
  pub fn coord_iter(&self) -> impl ExactSizeIterator<Item = CoordsRef<'_, S>> {
    self.vertices.column_iter().map(CoordsRef::new)
  }

  pub fn base_vertex(&self) -> CoordsRef<'_, S> {
    self.coord(0)
  }

  pub fn spanning_vector(&self, i: usize) -> Vector {
    assert!(i < self.dim_intrinsic());
    self.coord(i + 1) - self.base_vertex()
  }
  /// The spanning vectors $A$ of the parametrization, as the columns of a
  /// (space-by-intrinsic) matrix: its linear part.
  pub fn spanning_vectors(&self) -> Matrix {
    let mut mat = Matrix::zeros(self.dim_space().index(), self.dim_intrinsic().index());
    for i in 0..self.dim_intrinsic().index() {
      mat.set_column(i, &self.spanning_vector(i));
    }
    mat
  }

  pub fn det(&self) -> f64 {
    let a = self.spanning_vectors();
    // The signed volume factor: a determinant when the realization is square,
    // otherwise the Gram volume $sqrt(det(A^top A))$. Both are pure linear
    // algebra on the coordinates -- no inner product is supplied.
    let factor = if self.is_same_dim() {
      a.determinant()
    } else {
      (a.transpose() * &a).determinant().sqrt()
    };
    refsimp_vol(self.dim_intrinsic()) * factor
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
  /// A genuine inverse only when the realization is full-dimensional. On an
  /// embedded submanifold it is the Moore-Penrose one, which annihilates the
  /// normal space -- a metric-dependent choice, and hence not canonical.
  pub fn inv_linear_transform(&self) -> Matrix {
    if self.dim_intrinsic() == 0 {
      Matrix::zeros(0, 0)
    } else {
      self.linear_transform().pseudo_inverse(1e-12).unwrap()
    }
  }

  /// Local2Global Tangentvector
  pub fn pushforward_vector<'a>(&self, local: impl Into<VectorView<'a>>) -> Vector {
    self.linear_transform() * local.into()
  }
  /// Global2Local Cotangentvector
  pub fn pullback_covector<'a>(&self, global: impl Into<RowVectorView<'a>>) -> RowVector {
    global.into() * self.linear_transform()
  }

  pub fn affine_transform(&self) -> AffineTransform {
    let translation = self.base_vertex().view().into_owned();
    let linear = self.linear_transform();
    AffineTransform::new(translation, linear)
  }

  /// $psi_K$: the parametrization, from the reference chart out into the space.
  pub fn local2global<'a>(&self, local: impl Into<LocalRef<'a>>) -> Coords<S> {
    Coords::new(self.affine_transform().apply_forward(local.into().view()))
  }
  /// $psi_K^(-1)$: back from the space into the reference chart.
  pub fn global2local<'a>(&self, global: impl Into<CoordsRef<'a, S>>) -> Local {
    Local::new(self.affine_transform().apply_backward(global.into().view()))
  }
  pub fn global2bary<'a>(&self, global: impl Into<CoordsRef<'a, S>>) -> Bary {
    local2bary(&self.global2local(global))
  }
  pub fn bary2global<'a>(&self, bary: impl Into<BaryRef<'a>>) -> Coords<S> {
    let bary = bary.into();
    let global: Vector = self
      .coord_iter()
      .zip(bary.view().iter())
      .map(|(vi, &baryi)| baryi * vi.view())
      .sum();
    Coords::new(global)
  }

  /// Total differential of barycentric coordinate functions in the rows(!) of
  /// a matrix.
  pub fn difbarys(&self) -> Matrix {
    let difs = self.inv_linear_transform();
    let mut difs = difs.insert_row(0, 0.0);
    difs.set_row(0, &-difs.row_sum());
    difs
  }

  pub fn barycenter(&self) -> Coords<S> {
    let mut barycenter = Vector::zeros(self.dim_space().index());
    self.coord_iter().for_each(|v| barycenter += v.view());
    barycenter /= self.nvertices() as f64;
    Coords::new(barycenter)
  }
  pub fn is_global_inside<'a>(&self, global: impl Into<CoordsRef<'a, S>>) -> bool {
    is_bary_inside(&self.global2bary(global))
  }

  /// Coordinate subsimplices: each face of the simplex, realized in the same
  /// space by selecting its vertices' columns.
  pub fn subsimps(&self, sub_dim: Dim) -> impl Iterator<Item = SimplexCoords<S>> + use<'_, S> {
    Simplex::standard(self.dim_intrinsic())
      .subsimps(sub_dim)
      .collect::<Vec<_>>()
      .into_iter()
      .map(|sub| {
        let cols: Vec<Vector> = sub
          .iter()
          .map(|v| self.coord(v).view().into_owned())
          .collect();
        SimplexCoords::new(Matrix::from_columns(&cols))
      })
  }
  pub fn edges(&self) -> impl Iterator<Item = SimplexCoords<S>> + use<'_, S> {
    self.subsimps(Dim::ONE)
  }

  pub fn swap_vertices(&mut self, icol: usize, jcol: usize) {
    self.vertices.swap_columns(icol, jcol);
  }
  pub fn flip_orientation(&mut self) {
    if self.nvertices() >= 2 {
      self.swap_vertices(0, 1);
    } else {
      warn!("Cannot flip SimplexCoords with less than 2 vertices.");
    }
  }
  pub fn flipped_orientation(mut self) -> Self {
    self.flip_orientation();
    self
  }
}

impl SimplexCoords<LocalCartesian> {
  /// The standard simplex: the coordinate realization of the reference cell,
  /// whose local coordinates *are* the cartesian coordinates of its own chart.
  pub fn standard(ndim: Dim) -> Self {
    Self::new(ref_vertices(ndim))
  }
}

// The derives would demand `S: Clone`/`Debug`, which a marker never is.
impl<S: CoordSpace> Clone for SimplexCoords<S> {
  fn clone(&self) -> Self {
    Self::new(self.vertices.clone())
  }
}
impl<S: CoordSpace> std::fmt::Debug for SimplexCoords<S> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("SimplexCoords")
      .field("space", &S::NAME)
      .field("vertices", &self.vertices)
      .finish()
  }
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::Dim;
  use crate::atlas::{ref_bary, ref_difbarys};

  use approx::assert_relative_eq;

  /// The standard simplex is the coordinate realization of the reference chart:
  /// its local coordinates *are* the barycentric-derived ones.
  #[test]
  fn standard_barys() {
    for dim in (0..=4usize).map(Dim::from) {
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
  /// reference ones -- which is what lets any form built from them use
  /// [`ref_difbarys`] and never touch coordinates.
  #[test]
  fn standard_difbarys() {
    for dim in (0..=4usize).map(Dim::from) {
      let computed = SimplexCoords::standard(dim).difbarys();
      assert_relative_eq!(computed, ref_difbarys(dim), epsilon = 1e-12);
    }
  }

  /// The parametrization and its inverse are mutually inverse on the chart.
  #[test]
  fn local_global_roundtrip() {
    for dim in (1..=3usize).map(Dim::from) {
      let simp = SimplexCoords::standard(dim);
      let local = Local::from_iterator(dim.index(), (0..dim.index()).map(|i| 0.1 * (i + 1) as f64));
      let global = simp.local2global(&local);
      assert_relative_eq!(
        simp.global2local(&global).vector(),
        local.vector(),
        epsilon = 1e-12
      );
    }
  }
}
