//! The affine realization of a simplex in a coordinate space.
//!
//! A [`SimplexCoords<S>`] is the vertex coordinates of a simplex in the space
//! `S`, and the map it carries is the affine parametrization $x |-> v_0 + A x$
//! from the reference chart $hat(K)$ of the cell into `S`. It is a
//! parametrization, not a chart: the chart runs the other way and is the
//! barycentric one, which exists on every geometry ([`crate::atlas`]).
//!
//! The space is a type parameter. The same
//! construction is used in two genuinely different spaces:
//!
//! - `SimplexCoords<Ambient>` is a cell embedded in $RR^N$, the extrinsic
//!   realization, whose induced metric and edge lengths (`regge::coord`) are
//!   the bridges down into the intrinsic layer.
//! - `SimplexCoords<LocalCartesian>` is a simplex realized in a chart's own
//!   cartesian frame $RR^n$. [`unit`](SimplexCoords::unit) is the
//!   reference cell itself ("its ambient coordinates are its local
//!   coordinates"), and a sub-simplex of a refinement is the child realized in
//!   its parent's frame, the map its metric is pulled back along.
//!
//! Everything here is affine and metric-free: it needs coordinates in some
//! space, never an inner product on that space. The metric a realization
//! induces is a `regge::coord` concern layered on the `Ambient`
//! instantiation, and it is the only part that presupposes an embedding.

use super::{
  Bary, BaryRef, Local, LocalCartesian, LocalRef, is_bary_inside, local2bary, unit_simplex_volume,
  unit_vertices,
};
use crate::Dim;
use crate::linalg::{Matrix, RowVector, RowVectorView, Vector, VectorView};
use crate::topology::simplex::unit_subsimps;

use coorder::{Ambient, CoordSpace, Coords, CoordsRef, affine::AffineTransform};

use std::marker::PhantomData;

/// The relative floor the volume of a realization must clear to count as
/// non-degenerate, as a fraction of the largest volume its spanning vectors
/// could enclose.
const DEGENERACY_FLOOR: f64 = 1e-12;

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
    // algebra on the coordinates, no inner product is supplied.
    let factor = if self.is_same_dim() {
      a.determinant()
    } else {
      (a.transpose() * &a).determinant().sqrt()
    };
    unit_simplex_volume(self.dim_intrinsic()) * factor
  }
  pub fn vol(&self) -> f64 {
    self.det().abs()
  }
  /// Whether the realization is degenerate: the spanning vectors are
  /// numerically dependent, so the parametrization fails to be injective and
  /// the simplex collapses onto a lower-dimensional affine subspace.
  ///
  /// The volume is compared against the largest one a simplex with spanning
  /// vectors of these lengths could have, which by Hadamard's inequality is
  /// reached exactly when they are orthogonal. The quotient is therefore a
  /// dimensionless number in $[0, 1]$, one on an orthogonal frame and zero
  /// exactly on a dependent one, so the predicate is invariant under scaling
  /// the simplex: degeneracy is a rank condition, never a size.
  pub fn is_degenerate(&self) -> bool {
    let spanning_volume: f64 = self
      .spanning_vectors()
      .column_iter()
      .map(|v| v.norm())
      .product();
    self.vol() <= DEGENERACY_FLOOR * unit_simplex_volume(self.dim_intrinsic()) * spanning_volume
  }

  /// The linear part $A$ of the parametrization: the differential
  /// $dif psi_K$, constant because the parametrization is affine.
  pub fn linear_transform(&self) -> Matrix {
    self.spanning_vectors()
  }
  /// The pseudo-inverse $A^+$ of the linear part: the differential of the chart.
  ///
  /// A genuine inverse only when the realization is full-dimensional. On an
  /// embedded submanifold it is the Moore-Penrose one, which annihilates the
  /// normal space, a metric-dependent choice, and hence not canonical.
  pub fn inv_linear_transform(&self) -> Matrix {
    self.chart_transform().linear
  }

  /// $dif psi_K v$: a tangent vector of the reference chart, pushed forward
  /// into `S`.
  pub fn pushforward_vector<'a>(&self, local: impl Into<VectorView<'a>>) -> Vector {
    self.linear_transform() * local.into()
  }
  /// $psi_K^* omega$: a covector on `S`, pulled back onto the reference chart.
  /// The other variance of the same differential.
  pub fn pullback_covector<'a>(&self, global: impl Into<RowVectorView<'a>>) -> RowVector {
    global.into() * self.linear_transform()
  }

  /// $psi_K$ as an affine map, typed by the two spaces it runs between: out of
  /// the chart's cartesian frame and into `S`.
  ///
  /// A parametrization, and the type says so: its inverse is the chart,
  /// [`Self::chart_transform`].
  pub fn affine_transform(&self) -> AffineTransform<LocalCartesian, S> {
    let translation = self.base_vertex().to_coords();
    let linear = self.linear_transform();
    AffineTransform::new(translation, linear)
  }
  /// $psi_K^(-1)$ as an affine map: the chart, out of `S` and into the chart's
  /// cartesian frame.
  ///
  /// Inverse to [`Self::affine_transform`] on the affine hull of the simplex,
  /// which for a full-dimensional realization is all of `S`.
  pub fn chart_transform(&self) -> AffineTransform<S, LocalCartesian> {
    self.affine_transform().pseudo_inverse()
  }

  /// $psi_K$: the parametrization, from the reference chart out into the space.
  pub fn local2global<'a>(&self, local: impl Into<LocalRef<'a>>) -> Coords<S> {
    self.affine_transform().apply_forward(local.into())
  }
  /// $psi_K^(-1)$: back from the space into the reference chart.
  pub fn global2local<'a>(&self, global: impl Into<CoordsRef<'a, S>>) -> Local {
    self.chart_transform().apply_forward(global.into())
  }
  pub fn global2bary<'a>(&self, global: impl Into<CoordsRef<'a, S>>) -> Bary {
    local2bary(&self.global2local(global))
  }
  /// The point with the given barycentric coordinates: the affine combination of
  /// the vertices weighted by them, which is what barycentric coordinates are.
  pub fn bary2global<'a>(&self, bary: impl Into<BaryRef<'a>>) -> Coords<S> {
    Coords::affine_combination(
      bary
        .into()
        .into_view()
        .iter()
        .copied()
        .zip(self.coord_iter()),
    )
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
    Coords::barycenter(self.coord_iter())
  }
  pub fn is_global_inside<'a>(&self, global: impl Into<CoordsRef<'a, S>>) -> bool {
    is_bary_inside(&self.global2bary(global))
  }

  /// Coordinate subsimplices: each face of the simplex, realized in the same
  /// space by selecting its vertices' columns.
  pub fn subsimps<D: Into<Dim>>(
    &self,
    sub_dim: D,
  ) -> impl Iterator<Item = SimplexCoords<S>> + use<'_, S, D> {
    unit_subsimps(self.dim_intrinsic(), sub_dim).map(|positions| {
      let cols: Vec<Vector> = positions
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
  /// Reverse the orientation, by transposing the first two vertices.
  ///
  /// The identity on a point, whose orientation group is trivial: there is no
  /// transposition to make and nothing to report.
  pub fn flip_orientation(&mut self) {
    if self.nvertices() >= 2 {
      self.swap_vertices(0, 1);
    }
  }
  pub fn flipped_orientation(mut self) -> Self {
    self.flip_orientation();
    self
  }
}

impl SimplexCoords<LocalCartesian> {
  /// The unit simplex: the coordinate realization of the reference cell,
  /// whose local coordinates are the cartesian coordinates of its own chart.
  pub fn unit(ndim: impl Into<Dim>) -> Self {
    Self::new(unit_vertices(ndim))
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
  use crate::atlas::{unit_bary, unit_difbarys};

  use approx::assert_relative_eq;

  /// The unit simplex is the coordinate realization of the reference chart:
  /// its local coordinates are the barycentric-derived ones.
  #[test]
  fn unit_barys() {
    for dim in (0..=4usize).map(Dim::from) {
      let simp = SimplexCoords::unit(dim);
      for pos in simp.coord_iter() {
        let local = Local::new(pos.view().into_owned());
        let computed = simp.global2bary(pos);
        for ibary in 0..simp.nvertices() {
          let expected = unit_bary(ibary, &local);
          assert_eq!(computed[ibary], expected);
        }
      }
    }
  }

  /// The barycentric differentials of the unit simplex are the metric-free
  /// reference ones, which is what lets any form built from them use
  /// [`unit_difbarys`] and never touch coordinates.
  #[test]
  fn unit_difbarys_agree() {
    for dim in (0..=4usize).map(Dim::from) {
      let computed = SimplexCoords::unit(dim).difbarys();
      assert_relative_eq!(computed, unit_difbarys(dim), epsilon = 1e-12);
    }
  }

  /// A single vertex realized in $RR^N$: its one barycentric coordinate is
  /// constantly $1$, so its differential is the zero covector of that space, and
  /// the chart out of the empty local frame is the zero map. The degenerate end
  /// of the range, on the same code as every other simplex.
  #[test]
  fn point_realized_in_ambient_space() {
    for ambient in 0..=3 {
      let point: SimplexCoords = SimplexCoords::new(Matrix::from_element(ambient, 1, 0.7));
      assert_eq!(point.dim_intrinsic(), Dim::ZERO);
      assert_eq!(point.inv_linear_transform().shape(), (0, ambient));
      assert_relative_eq!(point.difbarys(), Matrix::zeros(1, ambient));
      assert_relative_eq!(
        point.barycenter().vector(),
        &Vector::from_element(ambient, 0.7)
      );
    }
  }

  /// Degeneracy is a rank condition and not a size: a simplex scaled uniformly
  /// stays non-degenerate however small it gets, and one whose vertices fall
  /// onto a lower-dimensional subspace is caught however large.
  #[test]
  fn degeneracy_is_scale_invariant() {
    for dim in (1..=4usize).map(Dim::from) {
      for scale in [1e-5, 1.0, 1e5] {
        let scaled = SimplexCoords::unit(dim).vertices() * scale;
        assert!(!SimplexCoords::<LocalCartesian>::new(scaled.clone()).is_degenerate());

        let mut collapsed = scaled;
        let base = collapsed.column(0).into_owned();
        collapsed.set_column(dim.index(), &base);
        assert!(SimplexCoords::<LocalCartesian>::new(collapsed).is_degenerate());
      }
    }
  }

  /// The parametrization and its inverse are mutually inverse on the chart.
  #[test]
  fn local_global_roundtrip() {
    for dim in (1..=3usize).map(Dim::from) {
      let simp = SimplexCoords::unit(dim);
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
