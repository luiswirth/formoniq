//! [`SimplexCoords`]'s local realization: barycentric coordinates agree with
//! [`unit_bary`], its differentials with [`unit_difbarys`], and degeneracy is
//! decided by the volume.

use approx::assert_relative_eq;
use simplicial::Dim;
use simplicial::atlas::{Local, LocalCartesian, SimplexCoords, unit_bary, unit_difbarys};
use simplicial::linalg::{Matrix, Vector};

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
