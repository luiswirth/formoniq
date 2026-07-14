//! The piecewise-affine atlas of the simplicial manifold.
//!
//! The cells of the complex are an atlas: each cell is a chart, and the
//! transition maps between overlapping cells are the affine gluings of the
//! shared faces ([`Transition`]). The atlas is therefore **piecewise affine**
//! -- a statement about the maps, which needs no metric. (Give it one, and the
//! simplicial manifold it presents is piecewise *flat*: curvature vanishes on
//! the cell interiors and concentrates on the codimension-2 hinges. That is a
//! statement about the geometry, and it is not this module's business.)
//!
//! A [`Chart`] *is* a cell -- top-dimensional by construction, since a face
//! carries no chart. A point of the simplicial manifold is thus intrinsically a
//! pair $(K, lambda)$ of a chart and the barycentric coordinates within it -- a
//! [`MeshPoint`]. Integration over a cell is quadrature over its chart, whose
//! nodes are such points ([`SimplexQuadRule`]).
//!
//! The chart's own structure -- the reference vertices, the barycentric
//! differentials, the volume -- depends on the dimension alone and not on the
//! cell: every chart of the atlas is the *same* chart up to the labelling of its
//! vertices. That is why the `ref_*` functions below take a [`Dim`] and no cell,
//! and it is why the element matrices of FEEC are computed once on the reference
//! cell and reused on every cell of the mesh. What differs between charts is the
//! labelling, and the labelling is exactly what a [`Transition`] is made of.
//!
//! Barycentric is the right chart: it is symmetric in the vertices, affine, and
//! needs neither a metric nor an embedding. Everything in this module is pure
//! affine combinatorics of the reference simplex, which is why it sits below
//! both the coordinate (extrinsic) and the metric layer, not inside either.
//!
//! # The two coordinate systems of a chart
//!
//! A chart carries two coordinate systems, related by dropping the redundant
//! zeroth weight:
//!
//! - [`Bary`]: barycentric $lambda in RR^(n+1)$ with $sum_i lambda_i = 1$, the
//!   symmetric one;
//! - [`Local`]: cartesian $x in RR^n$ with $x_i = lambda_(i+1)$ and
//!   $lambda_0 = 1 - sum_i x_i$, the one in which the reference frame -- and
//!   hence the value of a section -- is expressed.
//!
//! Both are distinct from the [`Ambient`](common::coord::Ambient) coordinates of
//! an embedding, and the [`common::coord::CoordSpace`] tags keep the
//! three from being confused for one another.

pub mod chart;
pub mod point;
pub mod quadrature;
pub mod transition;

pub use chart::{Chart, ChartExt};
pub use point::{MeshPoint, BARY_EPS};
pub use quadrature::SimplexQuadRule;
pub use transition::Transition;

use crate::Dim;

use common::{
  combo::{factorial_f64, Combination},
  coord::{CoordSpace, Coords, CoordsRef},
  linalg::nalgebra::{Matrix, RowVector, Vector},
};

/// The barycentric coordinate space of a chart: the affine weights
/// $lambda in RR^(n+1)$, $sum_i lambda_i = 1$.
pub enum Barycentric {}
impl CoordSpace for Barycentric {
  const NAME: &'static str = "bary";
}

/// The cartesian coordinate space of a chart: $x in RR^n$, the reference frame
/// in which the value of a section at a [`MeshPoint`] is expressed.
pub enum LocalCartesian {}
impl CoordSpace for LocalCartesian {
  const NAME: &'static str = "local";
}

/// Barycentric coordinates within a cell: $n+1$ affine weights summing to one.
pub type Bary = Coords<Barycentric>;
pub type BaryRef<'a> = CoordsRef<'a, Barycentric>;

/// Local (cartesian) coordinates within a cell chart.
pub type Local = Coords<LocalCartesian>;
pub type LocalRef<'a> = CoordsRef<'a, LocalCartesian>;

/// The volume of the reference $n$-simplex, $1 \/ n!$.
///
/// A property of the chart, not of the geometry: it is the factor by which a
/// chart integral scales, and the metric enters only through the further factor
/// $sqrt(det g)$ (see [`cell_volume`](crate::geometry::cell_volume)).
pub fn refsimp_vol(dim: Dim) -> f64 {
  factorial_f64(dim).recip()
}

pub fn bary2local<'a>(bary: impl Into<BaryRef<'a>>) -> Local {
  Local::new(bary.into().view().rows_range(1..).into_owned())
}
pub fn local2bary<'a>(local: impl Into<LocalRef<'a>>) -> Bary {
  let local = local.into();
  let bary0 = 1.0 - local.view().sum();
  Bary::new(local.view().insert_row(0, bary0))
}

/// Whether the barycentric weights lie in the closed reference cell, rather
/// than in the affine extension of the chart beyond it.
///
/// The weights sum to one -- that is what makes them barycentric -- so the
/// closed cell is cut out by their nonnegativity alone, and the upper bound
/// $lambda_i <= 1$ is implied rather than tested. Nonnegativity is tested up to
/// [`BARY_EPS`], because the weights vanishing on a face are only ever
/// floating-point zero, and a point of a face is a point of the cell.
pub fn is_bary_inside<'a>(bary: impl Into<BaryRef<'a>>) -> bool {
  let bary = bary.into();
  debug_assert!(
    approx::relative_eq!(bary.view().sum(), 1.0, epsilon = 1e-9),
    "Barycentric weights must sum to one."
  );
  bary.view().iter().all(|&b| b >= -BARY_EPS)
}

pub fn barycenter_bary(dim: Dim) -> Bary {
  Bary::from_element(dim + 1, ((dim + 1) as f64).recip())
}
pub fn barycenter_local(dim: Dim) -> Local {
  Local::from_element(dim, ((dim + 1) as f64).recip())
}

/// The $i$-th barycentric coordinate function evaluated in local coordinates.
pub fn ref_bary<'a>(ivertex: usize, local: impl Into<LocalRef<'a>>) -> f64 {
  let local = local.into();
  assert!(ivertex <= local.dim());
  if ivertex == 0 {
    1.0 - local.view().sum()
  } else {
    local[ivertex - 1]
  }
}

/// The differential $dif lambda_i$ of a barycentric coordinate function, a
/// constant covector in the reference frame.
pub fn ref_difbary(dim: Dim, ivertex: usize) -> RowVector {
  assert!(ivertex <= dim);
  if ivertex == 0 {
    RowVector::from_element(dim, -1.0)
  } else {
    let mut v = RowVector::zeros(dim);
    v[ivertex - 1] = 1.0;
    v
  }
}

/// The differential of the barycentric coordinate map
/// $lambda: RR^n -> RR^(n+1)$ of the reference simplex: the rows are the
/// constant covectors $dif lambda_i$.
///
/// Metric-free, and the same for every cell -- this is the whole reason the
/// Whitney forms can be evaluated intrinsically, with no geometry at all.
pub fn ref_difbarys(dim: Dim) -> Matrix {
  let mut difbarys = Matrix::zeros(dim + 1, dim);
  difbarys.row_mut(0).fill(-1.0);
  for i in 0..dim {
    difbarys[(i + 1, i)] = 1.0;
  }
  difbarys
}

/// The local coordinates of the vertices of the reference $n$-simplex, as the
/// columns of $[0 | I_n]$: the origin and the standard basis.
pub fn ref_vertices(dim: Dim) -> Matrix {
  let mut vertices = Matrix::zeros(dim, dim + 1);
  for i in 0..dim {
    vertices[(i, i + 1)] = 1.0;
  }
  vertices
}

/// The spanning vectors $v_i = e_(p_i) - e_(p_0)$ of a face of the reference
/// cell, in the cell's reference frame, as the columns of an $n times k$ matrix.
///
/// Pure affine combinatorics of the local vertex positions: the face of a cell
/// needs no coordinates of its own, and on a manifold without an embedding
/// there are none to be had.
pub fn ref_face_spanning_vectors(cell_dim: Dim, positions: &Combination) -> Matrix {
  let vertices = ref_vertices(cell_dim);
  let base = vertices.column(positions.index_at(0));
  let mut spanning = Matrix::zeros(cell_dim, positions.card() - 1);
  for (i, position) in positions.iter().skip(1).enumerate() {
    spanning.set_column(i, &(vertices.column(position) - base));
  }
  spanning
}

/// The barycentric coordinates, within a cell, of a point given by its
/// barycentric coordinates on a face: scatter the face's weights onto the local
/// vertex positions of the face, zero elsewhere.
pub fn face_bary_to_cell_bary<'a>(
  cell_dim: Dim,
  positions: &Combination,
  face_bary: impl Into<BaryRef<'a>>,
) -> Bary {
  let face_bary = face_bary.into();
  assert_eq!(
    face_bary.dim(),
    positions.card(),
    "Wrong number of weights."
  );
  let mut bary = Vector::zeros(cell_dim + 1);
  for (i, position) in positions.iter().enumerate() {
    bary[position] = face_bary[i];
  }
  Bary::new(bary)
}

#[cfg(test)]
mod test {
  use super::*;

  use approx::assert_relative_eq;

  /// The two chart coordinate systems are mutually inverse.
  #[test]
  fn bary_local_roundtrip() {
    for dim in 0..=4 {
      let local = Local::from_iterator(dim, (0..dim).map(|i| 0.1 * (i + 1) as f64));
      let bary = local2bary(&local);
      assert_relative_eq!(bary.sum(), 1.0, epsilon = 1e-12);
      assert_relative_eq!(bary2local(&bary).vector(), local.vector(), epsilon = 1e-12);
    }
  }

  /// The rows of the barycentric differential are the individual $dif lambda_i$,
  /// and they sum to zero: $sum_i lambda_i = 1$ is constant.
  #[test]
  fn ref_difbarys_rows_are_difbary_and_sum_to_zero() {
    for dim in 0..=4 {
      let difbarys = ref_difbarys(dim);
      for ivertex in 0..=dim {
        assert_relative_eq!(
          difbarys.row(ivertex).into_owned(),
          ref_difbary(dim, ivertex)
        );
      }
      assert_relative_eq!(difbarys.row_sum(), RowVector::zeros(dim));
    }
  }

  /// The barycentric coordinate functions are dual to the reference vertices,
  /// $lambda_i (e_j) = delta_(i j)$, and the differentials are their gradients.
  #[test]
  fn ref_barys_are_dual_to_ref_vertices() {
    for dim in 0..=4 {
      let vertices = ref_vertices(dim);
      for (j, vertex) in vertices.column_iter().enumerate() {
        let local = Local::new(vertex.into_owned());
        for i in 0..=dim {
          let expected = f64::from(i == j);
          assert_relative_eq!(ref_bary(i, &local), expected, epsilon = 1e-12);
        }
      }
    }
  }

  /// A face's spanning vectors are the differences of the reference vertices it
  /// selects, and a point of the face has the face's weights on those positions.
  #[test]
  fn face_bary_scatters_onto_the_face() {
    let cell_dim = 3;
    for face_dim in 0..=cell_dim {
      for positions in common::combo::combinations(cell_dim + 1, face_dim + 1) {
        let face_bary = barycenter_bary(face_dim);
        let bary = face_bary_to_cell_bary(cell_dim, &positions, &face_bary);

        assert_relative_eq!(bary.sum(), 1.0, epsilon = 1e-12);
        for i in 0..=cell_dim {
          let on_face = positions.iter().any(|p| p == i);
          assert_eq!(bary[i] != 0.0, on_face);
        }

        // The face's barycenter, expressed in the cell chart, is the mean of the
        // reference vertices the face selects.
        let vertices = ref_vertices(cell_dim);
        let spanning = ref_face_spanning_vectors(cell_dim, &positions);
        assert_eq!(spanning.ncols(), face_dim);
        let mean = positions
          .iter()
          .map(|p| vertices.column(p).into_owned())
          .sum::<Vector>()
          / (face_dim + 1) as f64;
        assert_relative_eq!(bary2local(&bary).vector(), &mean, epsilon = 1e-12);
      }
    }
  }
}
