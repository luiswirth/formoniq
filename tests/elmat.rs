extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use formoniq::{
  fe,
  geometry::{ref_vol, GeometrySimplex},
};

#[test]
fn check_elmat_refd() {
  for dim in 1..=10 {
    let nvertices = dim + 1;
    let ndofs = nvertices;

    // This expected element matrix has been verified over and over.
    // It should be correct.
    let mut expected_elmat = na::DMatrix::zeros(ndofs, ndofs);
    expected_elmat[(0, 0)] = dim as i32;
    for i in 1..ndofs {
      expected_elmat[(i, 0)] = -1;
      expected_elmat[(0, i)] = -1;
      expected_elmat[(i, i)] = 1;
    }
    let expected_laplacian = expected_elmat.cast::<f64>() * ref_vol(dim);

    let ref_simp = GeometrySimplex::new_ref(dim);
    let computed_laplacian = fe::laplacian_neg_elmat_geo(&ref_simp);
    let diff = &computed_laplacian - &expected_laplacian;
    println!("Computed:\n{computed_laplacian:.3}");
    println!("Expected:\n{expected_laplacian:.3}");
    println!("Difference:\n{diff:.3}");
    assert!(
      diff.norm() < dim as f64 * 10.0 * f64::EPSILON,
      "Wrong Laplacian Elmat in d={dim}"
    );
  }
}
