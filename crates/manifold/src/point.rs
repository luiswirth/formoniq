//! A point of the simplicial manifold, in the barycentric chart.
//!
//! The cells of the complex are an atlas: each cell is a chart, and the
//! transition maps between overlapping cells are the affine gluings of shared
//! faces. The atlas is therefore **piecewise affine** -- a statement about the
//! maps, which needs no metric. (Give it one, and the simplicial manifold it
//! presents is piecewise *flat*: curvature vanishes on the cell interiors and
//! concentrates on the codimension-2 hinges. That is a statement about the
//! geometry, and it is not this module's business.)
//!
//! A point of the simplicial manifold is thus intrinsically a pair
//! $(K, lambda)$ of a cell and the barycentric coordinates within it -- a
//! [`MeshPoint`].
//!
//! Barycentric is the right chart: it is symmetric in the vertices, affine,
//! and needs neither a metric nor an embedding. Everything in this module is
//! pure affine combinatorics of the reference simplex, which is why it sits
//! below both the coordinate (extrinsic) and the metric layer, not inside
//! either.
//!
//! The two coordinate systems on a cell are related by dropping the redundant
//! zeroth coordinate: barycentric $lambda in RR^(n+1)$ with
//! $sum_i lambda_i = 1$, local $x in RR^n$ with $x_i = lambda_(i+1)$ and
//! $lambda_0 = 1 - sum_i x_i$.

use crate::{
  topology::handle::{SimplexIdx, SimplexRef},
  Dim,
};

use common::linalg::nalgebra::{Matrix, RowVector, Vector, VectorView};

/// Barycentric coordinates within a cell: $n+1$ affine weights summing to one.
pub type Bary = Vector;
pub type BaryRef<'a> = VectorView<'a>;

/// A point of the simplicial manifold: a cell together with the barycentric
/// coordinates of the point within it.
///
/// The intrinsic notion of a point, defined on any geometry -- coordinates,
/// Regge edge lengths or bare cell metrics alike. Points on a shared face have
/// more than one such representation, one per incident cell; they agree on
/// everything the transition maps preserve.
#[derive(Debug, Clone, PartialEq)]
pub struct MeshPoint {
  pub cell: SimplexIdx,
  pub bary: Bary,
}

impl MeshPoint {
  pub fn new(cell: SimplexIdx, bary: Bary) -> Self {
    assert_eq!(bary.len(), cell.dim() + 1, "Wrong number of barycentrics.");
    Self { cell, bary }
  }
  /// From the local (cartesian) coordinates of the cell chart.
  pub fn from_local<'a>(cell: SimplexIdx, local: impl Into<VectorView<'a>>) -> Self {
    Self::new(cell, local2bary(local))
  }
  /// The barycenter of a cell.
  pub fn barycenter(cell: SimplexIdx) -> Self {
    Self::new(cell, barycenter_bary(cell.dim()))
  }

  /// The dimension of the manifold, which is that of the containing cell.
  pub fn dim(&self) -> Dim {
    self.cell.dim()
  }
  pub fn bary(&self) -> BaryRef<'_> {
    self.bary.as_view()
  }
  /// The local (cartesian) coordinates of the cell chart.
  pub fn local(&self) -> Vector {
    bary2local(&self.bary)
  }
  /// Whether the point lies in the closed cell, rather than in the affine
  /// extension of the chart beyond it.
  pub fn is_inside(&self) -> bool {
    is_bary_inside(&self.bary)
  }
}

pub fn bary2local<'a>(bary: impl Into<BaryRef<'a>>) -> Vector {
  let bary = bary.into();
  bary.view_range(1.., ..).into()
}
pub fn local2bary<'a>(local: impl Into<VectorView<'a>>) -> Bary {
  let local = local.into();
  let bary0 = 1.0 - local.sum();
  local.insert_row(0, bary0)
}

pub fn is_bary_inside<'a>(bary: impl Into<BaryRef<'a>>) -> bool {
  let bary = bary.into();
  approx::assert_relative_eq!(bary.sum(), 1.0, epsilon = 1e-9);
  bary.iter().all(|&b| (0.0..=1.0).contains(&b))
}

pub fn barycenter_bary(dim: Dim) -> Bary {
  Bary::from_element(dim + 1, ((dim + 1) as f64).recip())
}
pub fn barycenter_local(dim: Dim) -> Vector {
  Vector::from_element(dim, ((dim + 1) as f64).recip())
}

/// The $i$-th barycentric coordinate function evaluated in local coordinates.
pub fn ref_bary<'a>(ivertex: usize, local: impl Into<VectorView<'a>>) -> f64 {
  let local = local.into();
  assert!(ivertex <= local.len());
  if ivertex == 0 {
    1.0 - local.sum()
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

/// Geometry-free helpers on a cell handle.
pub trait CellPointExt {
  /// The barycenter of this cell, as a point of the manifold.
  fn barycenter_point(&self) -> MeshPoint;
}
impl CellPointExt for SimplexRef<'_> {
  fn barycenter_point(&self) -> MeshPoint {
    MeshPoint::barycenter(self.idx())
  }
}

#[cfg(test)]
mod test {
  use super::*;

  use approx::assert_relative_eq;

  /// The two chart coordinate systems are mutually inverse.
  #[test]
  fn bary_local_roundtrip() {
    for dim in 0..=4 {
      let local = Vector::from_fn(dim, |i, _| 0.1 * (i + 1) as f64);
      let bary = local2bary(&local);
      assert_relative_eq!(bary.sum(), 1.0, epsilon = 1e-12);
      assert_relative_eq!(bary2local(&bary), local, epsilon = 1e-12);
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
}
