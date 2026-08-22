//! The reference element matrices against their closed forms, swept over
//! every dimension the closed form is stated for: the anchors the whole
//! assembly rests on.
//!
//! Every chart of the atlas is the reference cell up to the labelling of its
//! vertices, so these matrices are what every assembled operator is built
//! from, and a closed form is the only oracle outside the library itself.

extern crate nalgebra as na;

use formoniq::{galerkin::BilinearForm, operators::WhitneyPairing};
use regge::lengths::simplex::SimplexLengthsSq;
use simplicial::linalg::Matrix;
use simplicial::{Dim, atlas::unit_simplex_volume, topology::complex::Complex};

use approx::assert_relative_eq;

/// The grade-$0$ stiffness $(dif lambda_i, dif lambda_j) |hat(K)|$: the
/// matrix with $n$ in the corner, $1$ on the remaining diagonal and $-1$
/// along the first row and column, scaled by the reference volume.
fn unit_laplacian(dim: Dim) -> Option<Matrix> {
  let ndofs = (dim + 1).index();
  let mut elmat = Matrix::zeros(ndofs, ndofs);
  elmat[(0, 0)] = dim.index() as f64;
  for i in 1..ndofs {
    elmat[(i, 0)] = -1.0;
    elmat[(0, i)] = -1.0;
    elmat[(i, i)] = 1.0;
  }
  Some(elmat * unit_simplex_volume(dim))
}

/// The grade-$0$ mass, the classical
/// $|hat(K)| (1 + delta_(i j)) \/ ((n + 1)(n + 2))$, tabulated for $n <= 3$.
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

/// The grade-$1$ Whitney mass, the anchor beside the grade-$0$ ones: on the
/// unit interval the single Whitney 1-form is $dif lambda_1$ of unit norm,
/// and on the unit triangle the hand-computed
/// $"diag"(1\/3, 1\/3, 1\/6)$ with the $1\/6$ coupling of the two edges at
/// the corner.
fn unit_mass_grade1(dim: Dim) -> Option<Matrix> {
  #[rustfmt::skip]
  let mats = [
    na::dmatrix![1.0],
    na::dmatrix![
      1./3., 1./6., 0.   ;
      1./6., 1./3., 0.   ;
      0.   , 0.   , 1./6.;
    ],
  ];
  mats.get(dim.index().checked_sub(1)?).cloned()
}

/// Every reference element matrix equals its closed form.
#[test]
fn the_reference_element_matrices_are_their_closed_forms() {
  type Anchor = (
    &'static str,
    fn(Dim) -> WhitneyPairing,
    fn(Dim) -> Option<Matrix>,
  );
  let anchors: [Anchor; 3] = [
    (
      "grade-0 stiffness",
      |dim| WhitneyPairing::dif_both(dim, 1),
      unit_laplacian,
    ),
    (
      "grade-0 mass",
      |dim| WhitneyPairing::mass(dim, Dim::ZERO),
      unit_mass,
    ),
    (
      "grade-1 mass",
      |dim| WhitneyPairing::mass(dim, 1),
      unit_mass_grade1,
    ),
  ];

  for (name, elmat, closed_form) in anchors {
    for dim in (1..=10).map(Dim::from) {
      let Some(expected) = closed_form(dim) else {
        continue;
      };

      let refcell = SimplexLengthsSq::unit(dim);
      let refcomplex = Complex::unit(dim);
      let refchart = refcomplex.cells().handle_iter().next().unwrap();
      let computed = elmat(dim).element(&refcell.metric(), refchart);

      assert_relative_eq!(&computed, &expected, epsilon = 1e-12);
      assert!(computed.norm() > 0.0, "{name} at dim {dim} is empty");
    }
  }
}
