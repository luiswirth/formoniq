extern crate nalgebra as na;

use common::linalg::nalgebra::Matrix;
use formoniq::operators::{self, ElMatProvider};
use manifold::{
  geometry::{metric::simplex::SimplexLengths, refsimp_vol},
  Dim,
};

use approx::assert_relative_eq;

fn check_ref_elmat<F, G, E>(elmat: G, ref_elmat: F)
where
  E: ElMatProvider,
  F: Fn(Dim) -> Option<Matrix>,
  G: Fn(Dim) -> E,
{
  for dim in 1..=10 {
    let Some(expected_elmat) = ref_elmat(dim) else {
      continue;
    };
    let elmat = elmat(dim);

    let refcell = SimplexLengths::standard(dim);
    let computed_elmat = elmat.eval(&refcell);

    assert_relative_eq!(&computed_elmat, &expected_elmat);
  }
}

#[test]
fn laplacian_refcell() {
  check_ref_elmat(operators::LaplaceBeltramiElmat::new, ref_laplacian);
}
fn ref_laplacian(dim: Dim) -> Option<Matrix> {
  let ndofs = dim + 1;
  let mut expected_elmat = Matrix::zeros(ndofs, ndofs);
  expected_elmat[(0, 0)] = dim as i32;
  for i in 1..ndofs {
    expected_elmat[(i, 0)] = -1;
    expected_elmat[(0, i)] = -1;
    expected_elmat[(i, i)] = 1;
  }

  Some(expected_elmat.cast::<f64>() * refsimp_vol(dim))
}

#[test]
fn mass_refcell() {
  check_ref_elmat(|_| operators::ScalarMassElmat, ref_mass);
}
fn ref_mass(dim: Dim) -> Option<Matrix> {
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
  mats.get(dim).cloned()
}

#[test]
fn lumped_mass_refcell() {
  check_ref_elmat(|_| operators::ScalarLumpedMassElmat, ref_lumped_mass);
}
fn ref_lumped_mass(dim: Dim) -> Option<Matrix> {
  let nvertices = dim + 1;
  let ndofs = nvertices;
  Some(refsimp_vol(dim) / ndofs as f64 * Matrix::identity(ndofs, ndofs))
}
