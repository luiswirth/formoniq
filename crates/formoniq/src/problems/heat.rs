//! Module for the Heat Equation, the prototypical parabolic PDE.

use common::linalg::faer::FaerCholesky;

use crate::{
  assemble,
  operators::{self, DofIdx},
};

use {
  manifold::{geometry::metric::MeshEdgeLengths, topology::complex::Complex},
  whitney::cochain::Cochain,
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
  let mass_csr = nas::CsrMatrix::from(&mass);
  let mut source = &mass_csr * &source_data.coeffs;

  assemble::enforce_dirichlet_bc(topology, &boundary_data, &mut laplace, &mut source);
  assemble::enforce_dirichlet_bc(topology, &boundary_data, &mut mass, &mut source);

  let laplace = nas::CsrMatrix::from(&laplace);
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
