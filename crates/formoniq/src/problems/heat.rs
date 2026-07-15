//! Module for the Heat Equation, the prototypical parabolic PDE.

use common::linalg::nalgebra::CsrMatrix;

use crate::{
  bc::LiftedSystem,
  whitney_complex::{BoundaryWhitneyComplex, WhitneyComplex},
};

use ddf::cochain::Cochain;
use exterior::ExteriorGrade;

/// Implicit Euler for the heat equation $diff_t u = -Delta u$ on Whitney
/// $k$-forms of any `grade`, with the essential boundary condition
/// $"tr" u = g$ held fixed in time (affine lifting, factorized once).
#[allow(clippy::too_many_arguments)]
pub fn solve_heat(
  whitney: WhitneyComplex,
  boundary: &BoundaryWhitneyComplex,
  grade: ExteriorGrade,
  nsteps: usize,
  dt: f64,
  boundary_values: &Cochain,
  initial_data: Cochain,
  source_data: Cochain,
  diffusion_coeff: f64,
) -> Vec<Cochain> {
  let laplace = CsrMatrix::from(&whitney.codif_dif(grade));
  let mass = CsrMatrix::from(&whitney.mass(grade));
  let system = &mass + diffusion_coeff * dt * &laplace;

  let lifted = LiftedSystem::new(&whitney.relative(), boundary, system, boundary_values);
  let source = &mass * source_data.coeffs();

  let mut solution = Vec::with_capacity(nsteps + 1);
  solution.push(initial_data);

  for _ in 0..nsteps {
    let prev = solution.last().unwrap().coeffs();
    let rhs = &mass * prev + dt * &source;
    solution.push(lifted.solve(&rhs));
  }

  solution
}
