//! Module for the Poisson Equation, the prototypical ellipitic PDE.

use common::util::FaerCholesky;
use manifold_complex::RiemannianComplex;

use crate::{assemble, fe, fe::DofIdx};

pub fn solve_poisson<F>(
  mesh: &RiemannianComplex,
  load_data: na::DVector<f64>,
  boundary_data: F,
) -> na::DVector<f64>
where
  F: Fn(DofIdx) -> f64,
{
  let elmat = fe::laplace_beltrami_elmat;
  let mut galmat = assemble::assemble_galmat(mesh, elmat);

  let elvec = fe::LoadElvec::new(load_data);
  let mut galvec = assemble::assemble_galvec(mesh, elvec);

  assemble::enforce_dirichlet_bc(mesh, boundary_data, &mut galmat, &mut galvec);

  let galmat = galmat.to_nalgebra_csc();
  FaerCholesky::new(galmat).solve(&galvec)
}
