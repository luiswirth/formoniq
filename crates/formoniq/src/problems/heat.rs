//! Module for the Heat Equation, the prototypical parabolic PDE.

use common::linalg::{faer::FaerCholesky, nalgebra::CsrMatrix};

use crate::{assemble, operators::DofIdx, whitney_complex::WhitneyComplex};

use ddf::cochain::Cochain;

/// times = [t_0,t_1,...,T]
pub fn solve_heat<F>(
  fes: WhitneyComplex,
  nsteps: usize,
  dt: f64,
  boundary_data: F,
  initial_data: Cochain,
  source_data: Cochain,
  diffusion_coeff: f64,
) -> Vec<Cochain>
where
  F: Fn(DofIdx) -> f64,
{
  let topology = fes.topology();

  let mut laplace = fes.codif_dif(0);
  let mut mass = fes.mass(0);
  let mass_csr = CsrMatrix::from(&mass);
  let mut source = &mass_csr * &source_data.coeffs;

  assemble::enforce_dirichlet_bc(topology, &boundary_data, &mut laplace, &mut source);
  assemble::enforce_dirichlet_bc(topology, &boundary_data, &mut mass, &mut source);

  let laplace = CsrMatrix::from(&laplace);
  let mass = mass_csr;

  let lse_matrix = &mass + diffusion_coeff * dt * &laplace;
  let lse_cholesky = FaerCholesky::new(lse_matrix);

  let mut solution = Vec::with_capacity(nsteps + 1);
  solution.push(initial_data);

  let last_step = nsteps - 1;
  for istep in 0..nsteps {
    println!("Solving Heat Equation at step={istep}/{last_step}...");

    let prev = solution.last().unwrap().coeffs();
    let rhs = &mass * prev + dt * &source;
    let next = lse_cholesky.solve(&rhs);

    solution.push(Cochain::new(0, next));
  }

  solution
}
