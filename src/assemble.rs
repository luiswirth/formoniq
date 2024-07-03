use nalgebra_sparse::CscMatrix;

use crate::{
  fe::{ElmatProvider, ElvecProvider},
  space::{DofId, FeSpace},
};

/// Assembly algorithm for the Galerkin Matrix in Lagrangian (0-form) FE.
pub fn assemble_galmat_lagrangian(
  space: &FeSpace,
  elmat: impl ElmatProvider,
) -> nas::CscMatrix<f64> {
  let mesh = space.mesh();
  let cell_dim = mesh.dim_intrinsic();

  // Lagrangian (0-form) has dofs associated with the nodes.
  let mut galmat = nas::CooMatrix::new(space.ndofs(), space.ndofs());
  for (icell, _) in mesh.dsimplicies(cell_dim).iter().enumerate() {
    let elmat = elmat.eval(mesh, (cell_dim, icell));
    for (ilocal, iglobal) in space
      .dof_indices_global((cell_dim, icell))
      .iter()
      .copied()
      .enumerate()
    {
      for (jlocal, jglobal) in space
        .dof_indices_global((cell_dim, icell))
        .iter()
        .copied()
        .enumerate()
      {
        galmat.push(iglobal, jglobal, elmat[(ilocal, jlocal)]);
      }
    }
  }
  nas::CscMatrix::from(&galmat)
}

/// Assembly algorithm for the Galerkin Vector in Lagrangian (0-form) FE.
pub fn assemble_galvec(space: &FeSpace, elvec: impl ElvecProvider) -> na::DVector<f64> {
  let mesh = space.mesh();
  let cell_dim = mesh.dim_intrinsic();

  let mut galvec = na::DVector::zeros(space.ndofs());
  for (icell, _) in mesh.dsimplicies(cell_dim).iter().enumerate() {
    let elvec = elvec.eval(mesh, (cell_dim, icell));
    for (ilocal, iglobal) in space
      .dof_indices_global((cell_dim, icell))
      .iter()
      .copied()
      .enumerate()
    {
      galvec[iglobal] += elvec[ilocal];
    }
  }
  galvec
}

/// Modifies supplied galerkin matrix and galerkin vector,
/// such that the FE solution has the optionally given coefficents on the dofs.
/// Is primarly used the enforce essential boundary conditions.
pub fn fix_dof_coeffs<F>(
  coefficent_map: F,
  galmat: &mut nas::CscMatrix<f64>,
  galvec: &mut na::DVector<f64>,
) where
  F: Fn(DofId) -> Option<f64>,
{
  let ndofs = galmat.ncols();

  // create vec of all (possibly missing) coefficents
  let dof_coeffs: Vec<_> = (0..ndofs).map(coefficent_map).collect();

  // zero out missing coefficents
  let dof_coeffs_zeroed =
    na::DVector::from_iterator(ndofs, dof_coeffs.iter().copied().map(|v| v.unwrap_or(0.0)));

  *galvec -= galmat.clone() * dof_coeffs_zeroed;

  // set galvec to prescribed coefficents
  dof_coeffs
    .iter()
    .copied()
    .enumerate()
    .filter_map(|(i, v)| v.map(|v| (i, v)))
    .for_each(|(i, v)| galvec[i] = v);

  // Set entires zero that share a (row or column) index with a fixed dof.
  let (trows, tcols, mut tvalues) = nas::CooMatrix::from(&*galmat).disassemble();
  for i in 0..trows.len() {
    let r = trows[i];
    let c = tcols[i];
    if dof_coeffs[r].is_some() || dof_coeffs[c].is_some() {
      // TODO: consider deleting triplet
      tvalues[i] = 0.0;
    }
  }
  let mut galmat_coo =
    nas::CooMatrix::try_from_triplets(ndofs, ndofs, trows, tcols, tvalues).unwrap();

  for (i, coeff) in dof_coeffs.iter().copied().enumerate() {
    if coeff.is_some() {
      galmat_coo.push(i, i, 1.0);
    }
  }

  *galmat = CscMatrix::from(&galmat_coo);
}

pub fn drop_dofs_galmat<F>(drop_map: F, galmat: &mut nas::CscMatrix<f64>)
where
  F: Fn(DofId) -> bool,
{
  let ndofs_old = galmat.ncols();

  let drop_ids: Vec<_> = (0..ndofs_old).filter(|idof| drop_map(*idof)).collect();
  let ndofs_new = ndofs_old - drop_ids.len();

  let (mut trows, mut tcols, mut tvalues) = nas::CooMatrix::from(&*galmat).disassemble();
  for (ndrops, mut drop_id) in drop_ids.iter().copied().enumerate() {
    drop_id -= ndrops;
    let mut itriplet = 0;
    while itriplet < trows.len() {
      let r = &mut trows[itriplet];
      let c = &mut tcols[itriplet];
      if *r == drop_id || *c == drop_id {
        trows.remove(itriplet);
        tcols.remove(itriplet);
        tvalues.remove(itriplet);
      } else {
        if *r > drop_id {
          *r -= 1;
        }
        if *c > drop_id {
          *c -= 1;
        }

        itriplet += 1;
      }
    }
  }
  let galmat_coo =
    nas::CooMatrix::try_from_triplets(ndofs_new, ndofs_new, trows, tcols, tvalues).unwrap();
  *galmat = nas::CscMatrix::from(&galmat_coo);
}

pub fn drop_dofs_galvec<F>(drop_map: F, galvec: &mut na::DVector<f64>)
where
  F: Fn(DofId) -> bool,
{
  let ndofs_old = galvec.ncols();
  let drop_ids: Vec<_> = (0..ndofs_old).filter(|idof| drop_map(*idof)).collect();
  let galvec_new = std::mem::replace(galvec, na::DVector::zeros(1));
  *galvec = galvec_new.remove_rows_at(&drop_ids);
}
