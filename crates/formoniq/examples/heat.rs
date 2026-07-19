//! Heat equation on the flat box $[0, pi]^n$: a parabolic flow *dissipates*.
//!
//! The method of lines --- Radau IIA in time over the Whitney space of
//! $k$-forms --- run at every dimension $1 <= n <= 3$ and grade $0 <= k <= n$
//! from one loop: the scalar heat equation at $k = 0$, the vector one at
//! $k = 1$, the top-form flow at $k = n$, all the same code.
//!
//! The operator is the full Hodge Laplacian $Delta = dif delta + delta dif$,
//! assembled in mixed form through the auxiliary $sigma = delta u$ (see
//! [`formoniq::problems::heat::solve_heat`]). It is symmetric positive
//! semidefinite, so the energy
//! $ E(t) = 1/2 norm(u)_(L^2)^2 $
//! is a Lyapunov functional: $dot(E) = -(u, Delta u) = -(norm(dif u)^2 +
//! norm(delta u)^2) <= 0$ for any initial state, and Radau IIA, being L-stable,
//! inherits the monotone decay unconditionally. The table shows $E$ falling
//! monotonically from a starting bump (a boundary-compatible eigenform, held at
//! zero on $diff K$ via the relative complex).
//!
//! At top grade $dif u = 0$ and $sigma = delta u$ is the only term; at grade
//! $0$ there is no $sigma$ and $Delta = delta dif$ --- both the trivial total
//! cases, run by the same code.

#[path = "util/mod.rs"]
mod util;

use {
  derham::{cochain::Cochain, project::derham_map, section::CoordFieldExt},
  formoniq::{
    linalg::quadratic_form_sparse, problems::heat::solve_heat, whitney_complex::WhitneyComplex,
  },
  simplicial::{
    gen::cartesian::CartesianMeshInfo,
    linalg::{CsrMatrix, Vector},
  },
  util::{BoundaryCondition, BoxEigenform},
};

use std::f64::consts::PI;

fn main() {
  tracing_subscriber::fmt::init();

  const NBOXES: usize = 8;
  const NSTEPS: usize = 40;
  const FINAL_TIME: f64 = 1.0;

  println!("Heat u_t = -Δu on [0,π]^n, relative (Dirichlet) BC — Radau IIA.");
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
    let metric = coords.to_edge_lengths_sq(&topology);
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
      let source = Cochain::new(grade, Vector::zeros(whitney.ndofs(grade)));

      // Homogeneous Dirichlet conditions are exactly the relative complex; the
      // starting eigenform already vanishes on $diff K$, so it lives there.
      let relative = whitney.relative();
      let dt = FINAL_TIME / NSTEPS as f64;
      let solution = solve_heat(&relative, grade, NSTEPS, dt, &initial, &source, 1.0);

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
