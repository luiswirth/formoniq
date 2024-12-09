use crate::{
  assemble, fe,
  mesh::SimplicialManifold,
  space::{DofIdx, FeSpace},
  sparse::SparseMatrix,
  util::{self, FaerCholesky},
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

  enforce_dirichlet_bc(mesh, boundary_data, &mut galmat, &mut galvec);

  let galmat = galmat.to_nalgebra_csc();

  FaerCholesky::new(galmat).solve(&galvec)
}

pub fn enforce_homogeneous_dirichlet_bc(
  mesh: &SimplicialManifold,
  galmat: &mut SparseMatrix,
  galvec: &mut na::DVector<f64>,
) {
  let boundary_nodes = mesh.boundary_nodes();
  fix_dofs_zero(&boundary_nodes, galmat, galvec);
}

pub fn enforce_dirichlet_bc<F>(
  mesh: &SimplicialManifold,
  boundary_coeff_map: F,
  galmat: &mut SparseMatrix,
  galvec: &mut na::DVector<f64>,
) where
  F: Fn(DofIdx) -> f64,
{
  let boundary_dofs = mesh.boundary_nodes();
  let dof_coeffs: Vec<_> = boundary_dofs
    .into_iter()
    .map(|idof| (idof, boundary_coeff_map(idof)))
    .collect();

  fix_dofs_coeff(&dof_coeffs, galmat, galvec);
}

pub fn fix_dofs_zero(dofs: &[DofIdx], galmat: &mut SparseMatrix, galvec: &mut na::DVector<f64>) {
  let ndofs = galmat.nrows();
  let dof_flags = util::indicies_to_flags(dofs, ndofs);
  galmat.set_zero(|i, j| dof_flags[i] || dof_flags[j]);
  for &idof in dofs {
    galmat.push(idof, idof, 1.0);
    galvec[idof] = 0.0;
  }
}

/// Fix DOFs of FE solution.
///
/// Is primarly used the enforce essential dirichlet boundary conditions.
///
/// Modifies supplied galerkin matrix and galerkin vector,
/// such that the FE solution has the optionally given coefficents on the dofs.
/// $mat(A_0, 0; 0, I) vec(mu_0, mu_diff) = vec(phi - A_(0 diff) gamma, gamma)$
pub fn fix_dofs_coeff(
  dof_coeffs: &[(DofIdx, f64)],
  galmat: &mut SparseMatrix,
  galvec: &mut na::DVector<f64>,
) {
  let ndofs = galmat.nrows();

  let dof_coeffs_opt = util::sparse_to_dense_data(dof_coeffs.to_vec(), ndofs);
  let dof_coeffs_zeroed =
    na::DVector::from_iterator(ndofs, dof_coeffs_opt.iter().map(|v| v.unwrap_or(0.0)));

  // Modify galvec.
  *galvec -= galmat.to_nalgebra_csc() * dof_coeffs_zeroed;

  // Set galvec to prescribed coefficents.
  dof_coeffs.iter().for_each(|&(i, v)| galvec[i] = v);

  // Set entires zero that share a (row or column) index with a fixed dof.
  galmat.set_zero(|r, c| dof_coeffs_opt[r].is_some() || dof_coeffs_opt[c].is_some());

  // Set galmat diagonal for dofs to one.
  for &(i, _) in dof_coeffs {
    galmat.push(i, i, 1.0);
  }
}

/// $mat(A_0, A_(0 diff); 0, I) vec(mu_0, mu_diff) = vec(phi, gamma)$
//#[allow(unused_variables, unreachable_code)]
pub fn fix_dofs_coeff_alt(
  dof_coeffs: &[(DofIdx, f64)],
  galmat: &mut SparseMatrix,
  galvec: &mut na::DVector<f64>,
) {
  tracing::warn!("use of `fix_dofs_coeff_alt` probably doesn't work.");

  let ndofs = galmat.nrows();
  let dof_coeffs_opt = util::sparse_to_dense_data(dof_coeffs.to_vec(), ndofs);

  // Set entires zero that share a row index with a fixed dof.
  galmat.set_zero(|r, _| dof_coeffs_opt[r].is_some());

  // Set galmat diagonal for dofs to one.
  for &(i, _) in dof_coeffs {
    galmat.push(i, i, 1.0);
  }

  // Set galvec to prescribed coefficents.
  for &(i, v) in dof_coeffs.iter() {
    galvec[i] = v
  }
}
