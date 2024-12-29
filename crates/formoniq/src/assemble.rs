use std::collections::HashSet;

use crate::fe::{DofIdx, ElmatProvider, ElvecProvider};

use common::{sparse::SparseMatrix, util};
use manifold::RiemannianComplex;

/// Assembly algorithm for the Galerkin Matrix.
pub fn assemble_galmat(mesh: &RiemannianComplex, elmat: impl ElmatProvider) -> SparseMatrix {
  let mut galmat = SparseMatrix::zeros(mesh.nvertices(), mesh.nvertices());
  for cell in mesh.cells() {
    let cell = cell.as_cell_complex();
    let faces = &cell.faces()[elmat.form_rank()];
    let elmat = elmat.eval(&cell);

    for (ilocal, &iglobal) in faces.iter().enumerate() {
      for (jlocal, &jglobal) in faces.iter().enumerate() {
        galmat.push(iglobal, jglobal, elmat[(ilocal, jlocal)]);
      }
    }
  }
  galmat
}

/// Assembly algorithm for the Galerkin Vector.
pub fn assemble_galvec(mesh: &RiemannianComplex, elvec: impl ElvecProvider) -> na::DVector<f64> {
  let mut galvec = na::DVector::zeros(mesh.nvertices());
  for cell in mesh.cells() {
    let cell = cell.as_cell_complex();
    let faces = &cell.faces()[elvec.form_rank()];
    let elvec = elvec.eval(&cell);
    for (ilocal, &iglobal) in faces.iter().enumerate() {
      galvec[iglobal] += elvec[ilocal];
    }
  }
  galvec
}

pub fn drop_boundary_dofs_galmat(mesh: &RiemannianComplex, galmat: &mut SparseMatrix) {
  drop_dofs_galmat(&mesh.boundary_vertices().into_iter().collect(), galmat)
}

pub fn drop_dofs_galmat(dofs: &HashSet<DofIdx>, galmat: &mut SparseMatrix) {
  assert!(galmat.nrows() == galmat.ncols());
  let ndofs_old = galmat.ncols();
  let ndofs_new = ndofs_old - dofs.len();

  let (_, _, triplets) = std::mem::take(galmat).into_parts();

  let mut triplets: Vec<_> = triplets
    .into_iter()
    .filter(|(r, c, _)| !dofs.contains(r) && !dofs.contains(c))
    .collect();

  for (r, c, _) in &mut triplets {
    let diffr = dofs.iter().filter(|&idof| idof < r).count();
    let diffc = dofs.iter().filter(|&idof| idof < c).count();
    *r -= diffr;
    *c -= diffc;
  }

  *galmat = SparseMatrix::new(ndofs_new, ndofs_new, triplets);
}

pub fn drop_dofs_galvec(dofs: &[DofIdx], galvec: &mut na::DVector<f64>) {
  *galvec = std::mem::take(galvec).remove_rows_at(dofs);
}

pub fn reintroduce_boundary_dofs_galsols(mesh: &RiemannianComplex, galsols: &mut na::DMatrix<f64>) {
  reintroduce_dropped_dofs_galsols(mesh.boundary_vertices(), galsols)
}

pub fn reintroduce_dropped_dofs_galsols(mut dofs: Vec<DofIdx>, galsols: &mut na::DMatrix<f64>) {
  dofs.sort_unstable();
  dofs.dedup();

  let mut galsol_owned = std::mem::take(galsols);
  for dof in dofs {
    galsol_owned = galsol_owned.insert_row(dof, 0.0);
  }
  *galsols = galsol_owned;
}

pub fn enforce_homogeneous_dirichlet_bc(
  mesh: &RiemannianComplex,
  galmat: &mut SparseMatrix,
  galvec: &mut na::DVector<f64>,
) {
  let boundary_vertices = mesh.boundary_vertices();
  fix_dofs_zero(&boundary_vertices, galmat, galvec);
}

pub fn enforce_dirichlet_bc<F>(
  mesh: &RiemannianComplex,
  boundary_coeff_map: F,
  galmat: &mut SparseMatrix,
  galvec: &mut na::DVector<f64>,
) where
  F: Fn(DofIdx) -> f64,
{
  let boundary_dofs = mesh.boundary_vertices();
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
  *galvec -= galmat.to_nalgebra_csr() * dof_coeffs_zeroed;

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
