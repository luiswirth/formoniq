//! Eigenvalue Problem

use crate::{
  assemble,
  fe::{self, ElmatProvider},
  mesh::SimplicialManifold,
  space::FeSpace,
};

use std::rc::Rc;

pub fn solve_homogeneous_evp(
  mesh: &Rc<SimplicialManifold>,
  operator_elmat: impl ElmatProvider,
) -> Vec<(f64, na::DVector<f64>)> {
  let space = FeSpace::new(Rc::clone(mesh));

  let mut operator_galmat = assemble::assemble_galmat(&space, operator_elmat);

  let mass_elmat = fe::lumped_mass_elmat;
  let mut mass_galmat = assemble::assemble_galmat(&space, mass_elmat);

  assemble::drop_boundary_dofs_galmat(mesh, &mut operator_galmat);
  assemble::drop_boundary_dofs_galmat(mesh, &mut mass_galmat);

  // solve generalized eigenvalue problem
  // $A u = lambda M u$
  unimplemented!()
}
