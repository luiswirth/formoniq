//! The Hodge wave equation: Gauss-Legendre conserves the full Hodge energy
//! to roundoff, at every dimension and grade.

use approx::assert_relative_eq;
use derham::Cochain;
use formoniq::problems::wave::{WaveState, solve_wave};
use formoniq::whitney_complex::{HilbertComplex, WhitneyComplex};
use regge::mesher::cartesian::CartesianGrid;
use simplicial::{Dim, linalg::Vector};

/// The hyperbolic law, at every dimension and grade: the full Hodge wave
/// energy is conserved to roundoff. Gauss-Legendre is symplectic and, on this
/// linear system, conserves the quadratic invariant exactly, through the
/// algebraic $sigma$ constraint and across the degenerate grades ($k = 0$: no
/// $sigma$; $k = n$: $dif u = 0$, energy is pure kinetic).
#[test]
fn energy_conserved_at_every_grade() {
  for dim in (2..=3).map(Dim::from) {
    let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
    let metric = coords.to_edge_lengths_sq(&topology);
    let whitney = WhitneyComplex::new(&topology, &metric);

    for grade in dim.range_inclusive() {
      let n = whitney.ndofs(grade);
      let pos = Vector::from_fn(n, |i, _| ((7 * i + 3) % 11) as f64 - 5.0);
      let vel = Vector::from_fn(n, |i, _| ((4 * i + 1) % 9) as f64 - 4.0);
      let force = Cochain::new(grade, Vector::zeros(n));

      let times: Vec<f64> = (0..=100).map(|i| 0.1 * i as f64).collect();
      let sol = solve_wave(&whitney, grade, &times, WaveState::new(pos, vel), force);

      let energy0 = sol[0].energy(&whitney, grade);
      for state in &sol {
        let energy = state.energy(&whitney, grade);
        assert_relative_eq!(energy, energy0, epsilon = 1e-8 * energy0.max(1.0));
      }
    }
  }
}
