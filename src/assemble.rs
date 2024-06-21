use std::rc::Rc;

use nalgebra_sparse::CscMatrix;

use crate::space::{DofId, FeSpace};

/// Assembly algorithm for the Galerkin Matrix in Lagrangian (0-form) FE.
/// `cell_dim`: The simplex dimension we are assembling over.
pub fn assemble_galmat_lagrangian(space: Rc<FeSpace>) -> nas::CscMatrix<f64> {
  let mesh = space.mesh();
  let cell_dim = mesh.dim_intrinsic();

  // Lagrangian (0-form) has dofs associated with the nodes.
  let mut galmat = nas::CooMatrix::new(space.ndofs(), space.ndofs());
  for (icell, _) in mesh.dsimplicies(cell_dim).iter().enumerate() {
    let cell_geo = mesh.coordinate_simplex((cell_dim, icell));
    let elmat = cell_geo.elmat();
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

  // set entires zero that share a (row or column) index with a fixed dof.
  let (mut trows, mut tcols, mut tvalues) = nas::CooMatrix::from(&*galmat).disassemble();
  let mut i = 0;
  while i < trows.len() {
    let r = trows[i];
    let c = tcols[i];
    if dof_coeffs[r].is_none() || dof_coeffs[c].is_none() {
      trows.remove(i);
      tcols.remove(i);
      tvalues.remove(i);
    } else {
      i += 1;
    }
  }
  let mut galmat_coo =
    nas::CooMatrix::try_from_triplets(ndofs, ndofs, trows, tcols, tvalues).unwrap();

  for (i, coeff) in dof_coeffs.iter().copied().enumerate() {
    galmat_coo.push(i, i, if coeff.is_some() { 1.0 } else { 0.0 });
  }

  *galmat = CscMatrix::from(&galmat_coo);
}
