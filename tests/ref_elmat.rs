extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use formoniq::{
  cell::{ref_vol, StandaloneCell},
  fe, Dim,
};

#[test]
fn elmat_refcell() {
  for dim in 1..=10 {
    let refcell = StandaloneCell::new_ref(dim);
    let computed_elmat = fe::laplacian_neg_elmat(&refcell);
    let expected_elmat = ref_elmat(dim);

    let diff = &computed_elmat - &expected_elmat;
    let error = diff.norm();
    let equal = error < 10e-12;
    if !equal {
      println!("Computed:\n{computed_elmat:.3}");
      println!("Expected:\n{expected_elmat:.3}");
      println!("Difference:\n{diff:.3}");
      panic!("Wrong reference elmat.");
    }
  }
}

fn ref_elmat(dim: Dim) -> na::DMatrix<f64> {
  let ndofs = dim + 1;
  let mut expected_elmat = na::DMatrix::zeros(ndofs, ndofs);
  expected_elmat[(0, 0)] = dim as i32;
  for i in 1..ndofs {
    expected_elmat[(i, 0)] = -1;
    expected_elmat[(0, i)] = -1;
    expected_elmat[(i, i)] = 1;
  }

  expected_elmat.cast::<f64>() * ref_vol(dim)
}
