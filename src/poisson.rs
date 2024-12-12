//! Module for the Poisson Equation, the prototypical ellipitic PDE.

use crate::{
  assemble, fe, lse,
  mesh::SimplicialManifold,
  space::{DofIdx, FeSpace},
  util::FaerCholesky,
};

use std::rc::Rc;

pub fn solve_poisson<F>(
  mesh: &Rc<SimplicialManifold>,
  load_data: na::DVector<f64>,
  boundary_data: F,
) -> na::DVector<f64>
where
  F: Fn(DofIdx) -> f64,
{
  let space = FeSpace::new(Rc::clone(mesh));

  let elmat = fe::laplace_beltrami_elmat;
  let mut galmat = assemble::assemble_galmat(&space, elmat);

  let elvec = fe::LoadElvec::new(load_data);
  let mut galvec = assemble::assemble_galvec(&space, elvec);

  lse::enforce_dirichlet_bc(mesh, boundary_data, &mut galmat, &mut galvec);

  let galmat = galmat.to_nalgebra_csc();
  FaerCholesky::new(galmat).solve(&galvec)
}
