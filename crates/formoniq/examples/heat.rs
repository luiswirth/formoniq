//! Heat equation on the flat box $[0, pi]^n$: a parabolic flow *dissipates*.
//!
//! The method of lines --- Radau IIA in time over the Whitney space of
//! $k$-forms --- run at every dimension $1 <= n <= 3$ and grade $0 <= k <= n$
//! from one loop: the scalar heat equation at $k = 0$, the vector one at
//! $k = 1$, the top-form flow at $k = n$, all the same code.
//!
//! The operator is the up-Laplacian $delta dif$ (the stiffness $D^T M D$),
//! which is the *full* Hodge Laplacian at grade $0$ and its assemblable part at
//! higher grade. It is symmetric positive semidefinite, so the energy
//! $ E(t) = 1/2 norm(u)_(L^2)^2 $
//! is a Lyapunov functional: $dot(E) = -(u, delta dif u) = -norm(dif u)^2 <= 0$
//! for any initial state, and Radau IIA, being L-stable, inherits the monotone
//! decay unconditionally. The table shows $E$ falling monotonically from any
//! starting bump (here a boundary-compatible eigenform, held at zero on
//! $diff K$).
//!
//! At top grade $dif u = 0$, so $delta dif = 0$ and the flow is static --- the
//! trivial total case, and why that row holds its energy rather than being
//! excluded.

#[path = "util/mod.rs"]
mod util;

use {
  common::linalg::nalgebra::{quadratic_form_sparse, CsrMatrix, Vector},
  ddf::{cochain::Cochain, derham::derham_map, section::CoordFieldExt},
  formoniq::{problems::heat::solve_heat, whitney_complex::WhitneyComplex},
  manifold::gen::cartesian::CartesianMeshInfo,
  util::{BoundaryCondition, BoxEigenform},
};

use std::f64::consts::PI;

fn main() {
  tracing_subscriber::fmt::init();

  const NBOXES: usize = 8;
  const NSTEPS: usize = 40;
  const FINAL_TIME: f64 = 1.0;

  println!("Heat u_t = -δd u on [0,π]^n, relative (Dirichlet) BC — Radau IIA.");
  println!("Energy E = ½‖u‖²_L² dissipates monotonically.\n");
  println!(
    "| {:>3} | {:>5} | {:>10} | {:>10} | {:>10} | {:>7} |",
    "dim", "grade", "E(0)", "E(½T)", "E(T)", "E(T)/E0",
  );

  for dim in 1..=3 {
    // Mesh and Whitney complex depend only on `dim`, not `grade`, so they are
    // built once per dimension and shared across the grade sweep below.
    let box_mesh = CartesianMeshInfo::new_unit_scaled(dim, NBOXES, PI);
    let (topology, coords) = box_mesh.compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);
    let whitney = WhitneyComplex::new(&topology, &metric);

    for grade in 0..=dim {
      let mass = CsrMatrix::from(&whitney.mass(grade));

      // A boundary-compatible starting bump: the relative eigenform vanishes on
      // $diff K$, reaching the mesh by pullback along the affine cell charts and
      // the de Rham map.
      let form = BoxEigenform::new(dim, grade, BoundaryCondition::Relative);
      let initial = derham_map(
        &form.solution().pullback_on(&topology, &coords),
        &topology,
        3,
      );
      let boundary = whitney.boundary().expect("the box has a boundary");
      let boundary_values = boundary.trace_cochain(&initial);
      let source = Cochain::new(grade, Vector::zeros(whitney.ndofs(grade)));

      let dt = FINAL_TIME / NSTEPS as f64;
      let solution = solve_heat(
        &whitney,
        &boundary,
        grade,
        NSTEPS,
        dt,
        &boundary_values,
        initial,
        source,
        1.0,
      );

      let energy = |c: &Cochain| 0.5 * quadratic_form_sparse(&mass, c.coeffs());
      let e0 = energy(&solution[0]);
      let e_half = energy(&solution[NSTEPS / 2]);
      let e_final = energy(solution.last().unwrap());
      println!(
        "| {dim:>3} | {grade:>5} | {e0:>10.3e} | {e_half:>10.3e} | {e_final:>10.3e} | {:>7.4} |",
        e_final / e0,
      );
    }
  }
}
