//! Laws for [`formoniq::whitney_complex`]: the codifferential is the
//! $L^2$-adjoint of the exterior derivative and is nilpotent, and the
//! element-local and global routes to an assembled operator agree.

use derham::Cochain;
use formoniq::whitney_complex::{HilbertComplex, WhitneyComplex};
use multialgebra::ExteriorGrade;
use regge::mesher::cartesian::CartesianGrid;
use simplicial::{Dim, linalg::Vector, topology::complex::Complex};

/// The stiffness is one matrix with two routes to it: the element-local
/// sandwich that [`HilbertComplex::dif_both`] assembles, and the global
/// product $D^T M_(k+1) D$ of three separately assembled matrices.
///
/// They agree because the exterior derivative of a Whitney form is the
/// coboundary of the reference cell, so the contraction commutes with the
/// scatter. Swept over every dimension and grade, the extremes included:
/// at the top grade both sides are the zero operator, which is the case a
/// route that special-cased the empty codomain would get wrong.
#[test]
fn dif_both_local_and_global_routes_agree() {
  for dim in (1..=3).map(Dim::from) {
    let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
    let lengths = coords.to_edge_lengths_sq(&topology);
    let whitney = WhitneyComplex::new(&topology, &lengths);

    for grade in dim.range_inclusive() {
      let local = whitney.dif_both(grade + 1);

      let dif = whitney.dif(grade);
      let mass = whitney.mass(grade + 1);
      let global = dif.transpose() * mass * dif;

      assert_eq!(local.nrows(), global.nrows());
      assert_eq!(local.ncols(), global.ncols());
      let residual = (&local - &global)
        .values()
        .iter()
        .fold(0.0f64, |acc, v| acc.max(v.abs()));
      let scale = local
        .values()
        .iter()
        .fold(0.0f64, |acc, v| acc.max(v.abs()));
      assert!(
        residual <= 1e-12 * scale.max(1.0),
        "dim {dim:?} grade {grade:?}: routes differ by {residual:e}"
      );
      // The law must be able to fail: at every grade below the top the
      // stiffness is a nonzero operator, so agreement is not two zeros
      // matching.
      if grade < dim {
        assert!(
          scale > 1e-6,
          "dim {dim:?} grade {grade:?}: stiffness vanished"
        );
      }
    }
  }
}

fn sample(grade: ExteriorGrade, topology: &Complex) -> Cochain {
  let ndofs = topology.nsimplices(grade);
  Cochain::new(
    grade,
    Vector::from_iterator(ndofs, (0..ndofs).map(|i| ((i * 3 % 7) as f64) - 3.0)),
  )
}

/// The defining law of the codifferential: it is the $L^2$-adjoint of $dif$,
/// $angle.l delta u, tau angle.r_(k-1) = angle.l u, dif tau angle.r_k$ for
/// every $tau in Lambda^(k-1)$. Swept over dimension and grade.
#[test]
fn codif_is_the_adjoint_of_dif() {
  use formoniq::linalg::bilinear_form_sparse;
  for dim in (1..=3).map(Dim::from) {
    let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
    let lengths = coords.to_edge_lengths_sq(&topology);
    let whitney = WhitneyComplex::new(&topology, &lengths);

    for grade in Dim::ONE.range_to_inclusive(dim) {
      let u = sample(grade, &topology);
      let tau = sample(grade - 1, &topology);
      let sigma = whitney.codif_cochain(&u);

      let mass_lower = whitney.mass(grade - 1);
      let mass_k = whitney.mass(grade);
      let lhs = bilinear_form_sparse(&mass_lower, sigma.coeffs(), tau.coeffs());
      let rhs = bilinear_form_sparse(&mass_k, u.coeffs(), tau.dif(&topology).coeffs());

      assert!(
        (lhs - rhs).abs() < 1e-9,
        "dim={dim} grade={grade}: {lhs} vs {rhs}"
      );
    }
  }
}

/// $delta compose delta = 0$: the codifferential is nilpotent, dual to
/// $dif compose dif = 0$. Needs grade $>= 2$ so both codifferentials land in a
/// real space.
#[test]
fn codif_is_nilpotent() {
  for dim in (2..=3).map(Dim::from) {
    let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
    let lengths = coords.to_edge_lengths_sq(&topology);
    let whitney = WhitneyComplex::new(&topology, &lengths);

    for grade in Dim::new(2).range_to_inclusive(dim) {
      let u = sample(grade, &topology);
      let ddu = whitney.codif_cochain(&whitney.codif_cochain(&u));
      assert!(whitney.norm_l2(&ddu) < 1e-9, "dim={dim} grade={grade}");
    }
  }
}
