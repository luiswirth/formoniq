//! The Hodge heat flow: with no source the $L^2$ energy can only decrease
//! (Radau IIA is L-stable), and its steady state reproduces the static
//! mixed Hodge-Laplace solution.

use approx::assert_relative_eq;
use derham::Cochain;
use formoniq::galerkin::GalerkinVector;
use formoniq::linalg::quadratic_form_sparse;
use formoniq::problems::elliptic::solve_source;
use formoniq::problems::heat::solve_heat;
use formoniq::whitney_complex::{HilbertComplex, WhitneyComplex};
use regge::mesher::cartesian::CartesianGrid;
use simplicial::{Dim, linalg::Vector};

/// The parabolic law, at every dimension and grade: with no source the
/// $L^2$ energy $norm(u)_M^2$ of the Hodge heat flow can only decrease.
/// $Delta$ is symmetric positive semidefinite, so the semidiscrete flow is a
/// contraction and Radau IIA, being L-stable, inherits it unconditionally.
/// The sweep exercises the degenerate grades too, $k = 0$ (no $sigma$) and
/// $k = n$ (no $omega$, $Delta = 0$, energy exactly flat).
#[test]
fn energy_dissipates_at_every_grade() {
  for dim in (1..=3).map(Dim::from) {
    let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
    let metric = coords.to_edge_lengths_sq(&topology);
    let whitney = WhitneyComplex::new(&topology, &metric);

    for grade in dim.range_inclusive() {
      let mass = whitney.mass(grade);
      let n = whitney.ndofs(grade);
      let u0 = Cochain::new(
        grade,
        Vector::from_fn(n, |i, _| ((5 * i + 2) % 7) as f64 - 3.0),
      );
      let source = Cochain::new(grade, Vector::zeros(n));

      let sol = solve_heat(&whitney, grade, 30, 0.05, &u0, &source, 1.0);

      let mut prev = f64::INFINITY;
      for u in &sol {
        let energy = quadratic_form_sparse(&mass, u.coeffs());
        assert!(
          energy <= prev + 1e-9,
          "energy must not increase (dim {dim}, grade {grade})"
        );
        prev = energy;
      }
    }
  }
}

/// The full Hodge Laplacian, not merely its up-part: the steady state of
/// $dot(u) = -Delta u + f$ must solve $Delta u = f$, i.e. reproduce the
/// independently assembled and factored static mixed Hodge-Laplace solution
/// [`solve_source`]. Run at grade $1$ on a topologically
/// trivial box (relative $b_1 = 0$, so the steady state is unique), where the
/// down-part $dif delta$ is genuinely nonzero, the two code paths agreeing
/// pins it down.
#[test]
fn steady_state_matches_static_hodge_laplace() {
  let (topology, coords) = CartesianGrid::new_unit(Dim::new(2), 3).triangulate();
  let metric = coords.to_edge_lengths_sq(&topology);
  let whitney = WhitneyComplex::new(&topology, &metric);
  let relative = whitney.relative();
  let grade = Dim::new(1);
  assert_eq!(relative.harmonic_dim(grade), 0);

  let n_rel = relative.ndofs(grade);
  let f_rel = Vector::from_fn(n_rel, |i, _| ((3 * i + 1) % 5) as f64 - 2.0);
  let inclusion = relative.inclusion(grade);
  let source = Cochain::new(grade, &inclusion * &f_rel);

  let mass_rel = relative.mass(grade);
  let galvec = GalerkinVector::new(grade, &inclusion * (&mass_rel * &f_rel));
  let (_sigma, u_static, _p) = solve_source(&relative, galvec, grade).expect("static solve");

  // Radau IIA's fixed point is exactly the steady state, so a large step
  // reaches it fast (and exactly, independent of dt).
  let zero = Cochain::new(grade, Vector::zeros(whitney.ndofs(grade)));
  let sol = solve_heat(&relative, grade, 200, 1.0, &zero, &source, 1.0);
  let u_final = sol.last().unwrap();

  assert_relative_eq!(u_final.coeffs(), u_static.coeffs(), epsilon = 1e-7);
}
