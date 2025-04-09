use crate::operators::{DofIdx, ElMatProvider, ElVecProvider};

use common::{linalg::nalgebra::CooMatrixExt, util};
use itertools::{multizip, Itertools};
use manifold::{geometry::metric::MeshEdgeLengths, topology::complex::Complex};

use rayon::prelude::*;
use std::collections::HashSet;

pub type GalMat = nas::CooMatrix<f64>;
pub type GalVec = na::DVector<f64>;

/// Assembly algorithm for the Galerkin Matrix.
pub fn assemble_galmat(
  topology: &Complex,
  geometry: &MeshEdgeLengths,
  elmat: impl ElMatProvider,
) -> GalMat {
  let row_grade = elmat.row_grade();
  let col_grade = elmat.col_grade();

  let nsimps_row = topology.skeleton(row_grade).len();
  let nsimps_col = topology.skeleton(col_grade).len();

  let triplets: Vec<(usize, usize, f64)> = topology
    .cells()
    .handle_iter()
    .par_bridge()
    .flat_map(|cell| {
      let geo = geometry.simplex_geometry(cell);
      let elmat = elmat.eval(&geo);

      let row_subs: Vec<_> = cell.subsimps(row_grade).collect();
      let col_subs: Vec<_> = cell.subsimps(col_grade).collect();

      let mut local_triplets = Vec::new();
      for (ilocal, &iglobal) in row_subs.iter().enumerate() {
        for (jlocal, &jglobal) in col_subs.iter().enumerate() {
          let val = elmat[(ilocal, jlocal)];
          if val != 0.0 {
            local_triplets.push((iglobal.kidx(), jglobal.kidx(), val));
          }
        }
      }

      local_triplets
    })
    .collect();

  let (rows, cols, values) = triplets.into_iter().multiunzip();
  GalMat::try_from_triplets(nsimps_row, nsimps_col, rows, cols, values).unwrap()
}

/// Assembly algorithm for the Galerkin Vector.
pub fn assemble_galvec(
  topology: &Complex,
  geometry: &MeshEdgeLengths,
  elvec: impl ElVecProvider,
) -> GalVec {
  let grade = elvec.grade();
  let nsimps = topology.skeleton(grade).len();

  let entries: Vec<(usize, f64)> = topology
    .cells()
    .handle_iter()
    .par_bridge()
    .flat_map(|cell| {
      let geo = geometry.simplex_geometry(cell);
      let elvec = elvec.eval(&geo);

      let subs: Vec<_> = cell.subsimps(grade).collect();

      let mut local_entires = Vec::new();
      for (ilocal, &iglobal) in subs.iter().enumerate() {
        if elvec[ilocal] != 0.0 {
          local_entires.push((iglobal.kidx(), elvec[ilocal]));
        }
      }

      local_entires
    })
    .collect();

  let mut galvec = na::DVector::zeros(nsimps);
  for (irow, val) in entries {
    galvec[irow] += val;
  }

  galvec
}

pub fn drop_boundary_dofs_galmat(complex: &Complex, galmat: &mut GalMat) {
  drop_dofs_galmat(&complex.boundary_vertices().kidx_iter().collect(), galmat)
}

pub fn drop_dofs_galmat(dofs: &HashSet<DofIdx>, galmat: &mut GalMat) {
  assert!(galmat.nrows() == galmat.ncols());
  let ndofs_old = galmat.ncols();
  let ndofs_new = ndofs_old - dofs.len();

  let (rows, cols, values) = std::mem::replace(galmat, GalMat::new(0, 0)).disassemble();

  let (rows, cols, values): (Vec<_>, Vec<_>, Vec<_>) = multizip((rows, cols, values))
    .filter(|(r, c, _)| !dofs.contains(r) && !dofs.contains(c))
    .map(|(mut r, mut c, v)| {
      let diffr = dofs.iter().filter(|&&idof| idof < r).count();
      let diffc = dofs.iter().filter(|&&idof| idof < c).count();
      r -= diffr;
      c -= diffc;
      (r, c, v)
    })
    .multiunzip();

  *galmat = GalMat::try_from_triplets(ndofs_new, ndofs_new, rows, cols, values).unwrap();
}

pub fn drop_dofs_galvec(dofs: &[DofIdx], galvec: &mut GalVec) {
  *galvec = std::mem::take(galvec).remove_rows_at(dofs);
}

pub fn reintroduce_boundary_dofs_galsols(complex: &Complex, galsols: &mut na::DMatrix<f64>) {
  reintroduce_dropped_dofs_galsols(complex.boundary_vertices().kidxs().to_vec(), galsols)
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
  complex: &Complex,
  galmat: &mut GalMat,
  galvec: &mut na::DVector<f64>,
) {
  let boundary_vertices = complex.boundary_vertices();
  fix_dofs_zero(boundary_vertices.kidxs(), galmat, galvec);
}

pub fn enforce_dirichlet_bc<F>(
  complex: &Complex,
  boundary_coeff_map: F,
  galmat: &mut GalMat,
  galvec: &mut na::DVector<f64>,
) where
  F: Fn(DofIdx) -> f64,
{
  let boundary_dofs = complex.boundary_vertices();
  let dof_coeffs: Vec<_> = boundary_dofs
    .kidx_iter()
    .map(|idof| (idof, boundary_coeff_map(idof)))
    .collect();

  fix_dofs_coeff(&dof_coeffs, galmat, galvec);
}

pub fn fix_dofs_zero(dofs: &[DofIdx], galmat: &mut GalMat, galvec: &mut na::DVector<f64>) {
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
  galmat: &mut GalMat,
  galvec: &mut na::DVector<f64>,
) {
  let ndofs = galmat.nrows();

  let dof_coeffs_opt = util::sparse_to_dense_data(dof_coeffs.to_vec(), ndofs);
  let dof_coeffs_zeroed =
    na::DVector::from_iterator(ndofs, dof_coeffs_opt.iter().map(|v| v.unwrap_or(0.0)));

  // Modify galvec.
  let galmat_csr = nas::CsrMatrix::from(&*galmat);
  *galvec -= galmat_csr * dof_coeffs_zeroed;

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
  galmat: &mut GalMat,
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
