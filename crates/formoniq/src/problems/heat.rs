//! Module for the Heat Equation, the prototypical parabolic PDE.

use crate::{
  assemble,
  operators::{self, DofIdx},
};

use common::util::FaerCholesky;
use manifold::{
  geometry::metric::MeshEdgeLengths,
  topology::complex::{attribute::Cochain, Complex},
};

/// times = [t_0,t_1,...,T]
#[allow(clippy::too_many_arguments)]
pub fn solve_heat<F>(
  topology: &Complex,
  geometry: &MeshEdgeLengths,
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
  let mut laplace = assemble::assemble_galmat(topology, geometry, operators::LaplaceBeltramiElmat);
  let mut mass = assemble::assemble_galmat(topology, geometry, operators::ScalarMassElmat);
  let mut source = mass.to_nalgebra_csr() * source_data.coeffs;

  assemble::enforce_dirichlet_bc(topology, &boundary_data, &mut laplace, &mut source);
  assemble::enforce_dirichlet_bc(topology, &boundary_data, &mut mass, &mut source);

  let laplace = laplace.to_nalgebra_csr();
  let mass = mass.to_nalgebra_csr();

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
