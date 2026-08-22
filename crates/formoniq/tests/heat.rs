//! The Hodge heat flow: with no source the $L^2$ energy can only decrease,
//! the parabolic law, which L-stable Radau IIA inherits unconditionally.

use derham::Cochain;
use formoniq::linalg::quadratic_form_sparse;
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
