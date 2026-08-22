//! The reference element matrices against their closed forms, swept over
//! every dimension $1 <= n <= 10$: the anchors the whole assembly rests on.

extern crate nalgebra as na;

use formoniq::{galerkin::BilinearForm, operators};
use regge::lengths::simplex::SimplexLengthsSq;
use simplicial::linalg::Matrix;
use simplicial::{Dim, atlas::unit_simplex_volume, topology::complex::Complex};

use approx::assert_relative_eq;

fn check_ref_elmat<F, G, E>(elmat: G, unit_elmat: F)
where
  E: BilinearForm,
  F: Fn(Dim) -> Option<Matrix>,
  G: Fn(Dim) -> E,
{
  for dim in (1..=10).map(Dim::from) {
    let Some(expected_elmat) = unit_elmat(dim) else {
      continue;
    };
    let elmat = elmat(dim);

    let refcell = SimplexLengthsSq::unit(dim);
    let refcomplex = Complex::unit(dim);
    let refchart = refcomplex.cells().handle_iter().next().unwrap();
    let computed_elmat = elmat.element(&refcell.metric(), refchart);

    assert_relative_eq!(&computed_elmat, &expected_elmat);
  }
}

/// The grade-$0$ stiffness of the reference $n$-simplex,
/// $(dif lambda_i, dif lambda_j) |hat(K)|$, against its closed form: the
/// matrix with $n$ in the corner, $1$ on the remaining diagonal and $-1$
/// along the first row and column, scaled by the reference volume.
#[test]
fn laplacian_refcell() {
  check_ref_elmat(
    |dim| operators::WhitneyPairing::dif_both(dim, 1),
    unit_laplacian,
  );
}
fn unit_laplacian(dim: Dim) -> Option<Matrix> {
  let ndofs = (dim + 1).index();
  let mut expected_elmat = Matrix::zeros(ndofs, ndofs);
  expected_elmat[(0, 0)] = dim.index() as i32;
  for i in 1..ndofs {
    expected_elmat[(i, 0)] = -1;
    expected_elmat[(0, i)] = -1;
    expected_elmat[(i, i)] = 1;
  }

  Some(expected_elmat.cast::<f64>() * unit_simplex_volume(dim))
}

/// The grade-$0$ mass of the reference $n$-simplex against the classical
/// $|hat(K)| (1 + delta_(i j)) \/ ((n + 1)(n + 2))$, tabulated for
/// $n <= 3$.
#[test]
fn mass_refcell() {
  check_ref_elmat(
    |dim| operators::WhitneyPairing::mass(dim, Dim::ZERO),
    unit_mass,
  );
}
fn unit_mass(dim: Dim) -> Option<Matrix> {
  #[rustfmt::skip]
  let mats = [
    na::dmatrix![1.0],
    na::dmatrix![
      1.0/3.0, 1.0/6.0;
      1.0/6.0, 1.0/3.0;
    ],
    na::dmatrix![
      1.0/12.0, 1.0/24.0, 1.0/24.0;
      1.0/24.0, 1.0/12.0, 1.0/24.0;
      1.0/24.0, 1.0/24.0, 1.0/12.0;
    ],
    na::dmatrix![
      1.0/60.0, 1.0/120.0, 1.0/120.0, 1.0/120.0;
      1.0/120.0, 1.0/60.0, 1.0/120.0, 1.0/120.0;
      1.0/120.0, 1.0/120.0, 1.0/60.0, 1.0/120.0;
      1.0/120.0, 1.0/120.0, 1.0/120.0, 1.0/60.0;
    ],
  ];
  mats.get(dim.index()).cloned()
}
