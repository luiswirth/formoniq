extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use common::{linalg::assert_mat_eq, Dim};
use formoniq::fe::{self, ElmatProvider};
use manifold::simplicial::{ref_vol, ReferenceCell};

fn check_ref_elmat<F>(elmat: impl ElmatProvider, ref_elmat: F)
where
  F: Fn(Dim) -> Option<na::DMatrix<f64>>,
{
  for dim in 1..=10 {
    let Some(expected_elmat) = ref_elmat(dim) else {
      continue;
    };

    let refcell = ReferenceCell::new(dim).to_cell_complex();
    let computed_elmat = elmat.eval(&refcell);

    assert_mat_eq(&computed_elmat, &expected_elmat);
  }
}

#[test]
fn laplacian_refcell() {
  check_ref_elmat(fe::laplace_beltrami_elmat, ref_laplacian);
}
fn ref_laplacian(dim: Dim) -> Option<na::DMatrix<f64>> {
  let ndofs = dim + 1;
  let mut expected_elmat = na::DMatrix::zeros(ndofs, ndofs);
  expected_elmat[(0, 0)] = dim as i32;
  for i in 1..ndofs {
    expected_elmat[(i, 0)] = -1;
    expected_elmat[(0, i)] = -1;
    expected_elmat[(i, i)] = 1;
  }

  Some(expected_elmat.cast::<f64>() * ref_vol(dim))
}

#[test]
fn mass_refcell() {
  check_ref_elmat(fe::mass_elmat, ref_mass);
}
fn ref_mass(dim: Dim) -> Option<na::DMatrix<f64>> {
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
  check_ref_elmat(fe::lumped_mass_elmat, ref_lumped_mass);
}
fn ref_lumped_mass(dim: Dim) -> Option<na::DMatrix<f64>> {
  let nvertices = dim + 1;
  let ndofs = nvertices;
  Some(ref_vol(dim) / ndofs as f64 * na::DMatrix::identity(ndofs, ndofs))
}