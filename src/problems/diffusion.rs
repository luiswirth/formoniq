//! Module for the (Heat) Diffussion Equation, the prototypical parabolic PDE.

use crate::{
  assemble,
  fe::{self, DofIdx},
  mesh::SimplicialManifold,
  util::FaerCholesky,
};

/// times = [t_0,t_1,...,T]
pub fn solve_diffusion<F>(
  mesh: &SimplicialManifold,
  nsteps: usize,
  dt: f64,
  boundary_data: F,
  initial_data: na::DVector<f64>,
  source_data: na::DVector<f64>,
  coeff: f64,
) -> Vec<na::DVector<f64>>
where
  F: Fn(DofIdx) -> f64,
{
  let mut laplace = assemble::assemble_galmat(mesh, fe::laplace_beltrami_elmat);
  let mut mass = assemble::assemble_galmat(mesh, fe::mass_elmat);
  let mut source = assemble::assemble_galvec(mesh, fe::LoadElvec::new(source_data));

  assemble::enforce_dirichlet_bc(mesh, &boundary_data, &mut laplace, &mut source);
  assemble::enforce_dirichlet_bc(mesh, &boundary_data, &mut mass, &mut source);

  let laplace = laplace.to_nalgebra_csc();
  let mass = mass.to_nalgebra_csc();

  let lse_matrix = &mass + coeff * dt * &laplace;
  let lse_cholesky = FaerCholesky::new(lse_matrix);

  let mut solution = Vec::with_capacity(nsteps + 1);
  solution.push(initial_data);

  let last_step = nsteps - 1;
  for istep in 0..nsteps {
    println!("Solving Diffusion Equation at step={istep}/{last_step}...");

    let prev = solution.last().unwrap();
    let rhs = &mass * prev + dt * &source;
    let next = lse_cholesky.solve(&rhs);

    solution.push(next);
  }

  solution
}
