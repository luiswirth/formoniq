//! Wave equation on the flat box $[0, pi]^n$: a hyperbolic flow *conserves*.
//!
//! Semi-discrete: the Whitney space of $k$-forms in space, Gauss-Legendre
//! (implicit midpoint) in time, run at every dimension $1 <= n <= 3$ and grade
//! $0 <= k <= n$ from one loop --- the scalar wave equation at $k = 0$, the
//! vector (curl-curl) one at $k = 1$, the top-form flow at $k = n$.
//!
//! The operator is the up-Laplacian $delta dif$ (the stiffness $K = D^T M D$),
//! the full Hodge Laplacian at grade $0$ and its assemblable part above. The
//! semi-discrete energy
//! $ E = 1/2 (u^T K u + dot(u)^T M dot(u)) $
//! --- potential plus kinetic --- is conserved: Gauss-Legendre is symplectic
//! and, applied to this linear system, conserves $E$ to roundoff. The table
//! shows $E$ holding across the run; the drift column is the largest relative
//! departure over all steps.
//!
//! At top grade $dif u = 0$, so $K = 0$ and the potential energy vanishes: from
//! rest the state is static at zero energy --- the trivial total case.

#[path = "util/mod.rs"]
mod util;

use {
  common::linalg::nalgebra::{CsrMatrix, Vector},
  ddf::{cochain::Cochain, derham::derham_map, section::CoordFieldExt},
  formoniq::{
    problems::wave::{cfl_dt, solve_wave, WaveState},
    whitney_complex::WhitneyComplex,
  },
  manifold::gen::cartesian::CartesianMeshInfo,
  util::{BoundaryCondition, BoxEigenform},
};

use std::f64::consts::{PI, TAU};

fn main() {
  tracing_subscriber::fmt::init();

  const NBOXES: usize = 8;
  const DURATION: f64 = TAU;
  // A conservative fraction of the CFL estimate: the mesh-width heuristic
  // overshoots the true leapfrog limit $dt thin omega_max < 2$ the more each
  // vertex is connected, i.e. in higher dimension.
  const CFL_FRACTION: f64 = 0.2;
  const WAVE_SPEED: f64 = 1.0;

  println!("Wave u_tt = -δd u on [0,π]^n, relative (Dirichlet) BC — Gauss-Legendre.");
  println!("Energy E = ½(uᵀKu + u̇ᵀMu̇) is conserved.\n");
  println!(
    "| {:>3} | {:>5} | {:>10} | {:>10} | {:>10} | {:>8} |",
    "dim", "grade", "E(0)", "E(½T)", "E(T)", "drift",
  );

  for dim in 1..=3 {
    // Mesh and Whitney complex depend only on `dim`, not `grade`, so they are
    // built once per dimension and shared across the grade sweep below.
    let box_mesh = CartesianMeshInfo::new_unit_scaled(dim, NBOXES, PI);
    let (topology, coords) = box_mesh.compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);
    let whitney = WhitneyComplex::new(&topology, &metric);

    for grade in 0..=dim {
      let laplace = CsrMatrix::from(&whitney.codif_dif(grade));
      let mass = CsrMatrix::from(&whitney.mass(grade));

      // Released from rest with a boundary-compatible bump: the relative
      // eigenform vanishes on $diff K$.
      let form = BoxEigenform::new(dim, grade, BoundaryCondition::Relative);
      let initial = derham_map(
        &form.solution().pullback_on(&topology, &coords),
        &topology,
        3,
      );
      let state = WaveState::new(initial.into_coeffs(), Vector::zeros(whitney.ndofs(grade)));

      let dt = CFL_FRACTION * cfl_dt(&metric, WAVE_SPEED);
      let nsteps = (DURATION / dt).ceil() as usize;
      let times: Vec<f64> = (0..=nsteps)
        .map(|i| DURATION * i as f64 / nsteps as f64)
        .collect();

      let force = Cochain::new(grade, Vector::zeros(whitney.ndofs(grade)));
      let solution = solve_wave(&whitney, grade, &times, state, force);

      let energies: Vec<f64> = solution.iter().map(|s| s.energy(&laplace, &mass)).collect();
      let e0 = energies[0];
      let e_half = energies[nsteps / 2];
      let e_final = *energies.last().unwrap();
      let drift = if e0 > 1e-14 {
        energies.iter().map(|&e| (e - e0).abs()).fold(0.0, f64::max) / e0
      } else {
        0.0
      };
      println!(
        "| {dim:>3} | {grade:>5} | {e0:>10.3e} | {e_half:>10.3e} | {e_final:>10.3e} | {drift:>8.1e} |",
      );
    }
  }
}
