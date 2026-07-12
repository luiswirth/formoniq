//! Module for the Heat Equation, the prototypical parabolic PDE.

use common::linalg::nalgebra::CsrMatrix;

use crate::{
  bc::LiftedSystem,
  whitney_complex::{BoundaryWhitneyComplex, WhitneyComplex},
};

use ddf::cochain::Cochain;

/// Implicit Euler for the heat equation, with the essential boundary
/// condition $"tr" u = g$ held fixed in time (affine lifting, factorized
/// once).
#[allow(clippy::too_many_arguments)]
pub fn solve_heat(
  fes: WhitneyComplex,
  boundary: &BoundaryWhitneyComplex,
  nsteps: usize,
  dt: f64,
  boundary_values: &Cochain,
  initial_data: Cochain,
  source_data: Cochain,
  diffusion_coeff: f64,
) -> Vec<Cochain> {
  let laplace = CsrMatrix::from(&fes.codif_dif(0));
  let mass = CsrMatrix::from(&fes.mass(0));
  let system = &mass + diffusion_coeff * dt * &laplace;

  let lifted = LiftedSystem::new(&fes.relative(), boundary, system, boundary_values);
  let source = &mass * &source_data.coeffs;

  let mut solution = Vec::with_capacity(nsteps + 1);
  solution.push(initial_data);

  let last_step = nsteps - 1;
  for istep in 0..nsteps {
    println!("Solving Heat Equation at step={istep}/{last_step}...");

    let prev = solution.last().unwrap().coeffs();
    let rhs = &mass * prev + dt * &source;
    solution.push(lifted.solve(&rhs));
  }

  solution
}
