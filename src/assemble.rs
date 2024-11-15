use crate::{
  fe::{ElmatProvider, ElvecProvider},
  matrix::SparseMatrix,
  space::{DofId, FeSpace},
  util::{faervec2navec, navec2faervec},
};

/// Assembly algorithm for the Galerkin Matrix.
pub fn assemble_galmat(space: &FeSpace, elmat: impl ElmatProvider) -> SparseMatrix {
  let mut galmat = SparseMatrix::new(space.ndofs(), space.ndofs());
  for icell in 0..space.mesh().ncells() {
    let elmat = elmat.eval(space, icell);
    println!("{elmat:.3}");

    for (ilocal, iglobal) in space
      .dof_handler()
      .local2global(icell)
      .iter()
      .copied()
      .enumerate()
    {
      for (jlocal, jglobal) in space
        .dof_handler()
        .local2global(icell)
        .iter()
        .copied()
        .enumerate()
      {
        galmat.push(iglobal, jglobal, elmat[(ilocal, jlocal)]);
      }
    }
  }
  galmat
}

/// Assembly algorithm for the Galerkin Vector.
pub fn assemble_galvec(space: &FeSpace, elvec: impl ElvecProvider) -> na::DVector<f64> {
  let mut galvec = na::DVector::zeros(space.ndofs());
  for icell in 0..space.mesh().ncells() {
    let elvec = elvec.eval(space, icell);
    for (ilocal, iglobal) in space
      .dof_handler()
      .local2global(icell)
      .iter()
      .copied()
      .enumerate()
    {
      galvec[iglobal] += elvec[ilocal];
    }
  }
  galvec
}

pub trait DofCoeffMap {
  fn eval(&self, idof: DofId) -> Option<f64>;
}
impl<F> DofCoeffMap for F
where
  F: Fn(DofId) -> Option<f64>,
{
  fn eval(&self, idof: DofId) -> Option<f64> {
    self(idof)
  }
}

/// Fix DOFs of FE solution.
///
/// Is primarly used the enforce essential dirichlet boundary conditions.
///
/// Modifies supplied galerkin matrix and galerkin vector,
/// such that the FE solution has the optionally given coefficents on the dofs.
/// $mat(A_0, 0; 0, I) vec(mu_0, mu_diff) = vec(phi - A_(0 diff) gamma, gamma)$
pub fn fix_dof_coeffs<F>(
  coefficent_map: F,
  galmat: &mut SparseMatrix,
  galvec: &mut na::DVector<f64>,
) where
  F: DofCoeffMap,
{
  let ndofs = galmat.ncols();

  // create vec of all (possibly missing) coefficents
  let dof_coeffs: Vec<_> = (0..ndofs).map(|idof| coefficent_map.eval(idof)).collect();

  // zero out missing coefficents
  let mut dof_coeffs_zeroed = faer::Mat::zeros(ndofs, 1);
  dof_coeffs
    .iter()
    .copied()
    .map(|v| v.unwrap_or(0.0))
    .enumerate()
    .for_each(|(i, v)| dof_coeffs_zeroed[(i, 0)] = v);

  let galmat_faer = galmat.to_faer_csc();
  let mut galvec_faer = navec2faervec(galvec);

  galvec_faer -= galmat_faer * dof_coeffs_zeroed;
  *galvec = faervec2navec(&galvec_faer);

  // set galvec to prescribed coefficents
  dof_coeffs
    .iter()
    .copied()
    .enumerate()
    .filter_map(|(i, v)| v.map(|v| (i, v)))
    .for_each(|(i, v)| galvec[i] = v);

  // Set entires zero that share a (row or column) index with a fixed dof.
  galmat.set_zero(|r, c| dof_coeffs[r].is_some() || dof_coeffs[c].is_some());

  for (i, coeff) in dof_coeffs.iter().copied().enumerate() {
    if coeff.is_some() {
      galmat.push(i, i, 1.0);
    }
  }
}

/// $mat(A_0, A_(0 diff); 0, I) vec(mu_0, mu_diff) = vec(phi, gamma)$
#[allow(unused_variables, unreachable_code)]
pub fn fix_dof_coeffs_alt<F>(
  coefficent_map: F,
  galmat: &mut SparseMatrix,
  galvec: &mut na::DVector<f64>,
) where
  F: Fn(DofId) -> Option<f64>,
{
  panic!("DOES NOT WORK");

  let ndofs = galmat.ncols();

  // create vec of all (possibly missing) coefficents
  let dof_coeffs: Vec<_> = (0..ndofs).map(coefficent_map).collect();

  // set galvec to prescribed coefficents
  dof_coeffs
    .iter()
    .copied()
    .enumerate()
    .filter_map(|(i, v)| v.map(|v| (i, v)))
    .for_each(|(i, v)| galvec[i] = v);

  // Set entires zero that share a row index with a fixed dof.
  galmat.set_zero(|r, c| dof_coeffs[r].is_some());

  for (i, coeff) in dof_coeffs.iter().copied().enumerate() {
    galmat.push(i, i, 1.0);
  }
}

pub fn drop_dofs_galmat<F>(drop_map: F, galmat: &mut SparseMatrix)
where
  F: Fn(DofId) -> bool,
{
  let ndofs_old = galmat.ncols();

  let drop_ids: Vec<_> = (0..ndofs_old).filter(|idof| drop_map(*idof)).collect();
  let ndofs_new = ndofs_old - drop_ids.len();

  let mut triplets = std::mem::take(galmat).to_triplets();
  for (ndrops, mut drop_id) in drop_ids.iter().copied().enumerate() {
    drop_id -= ndrops;
    let mut itriplet = 0;
    while itriplet < triplets.len() {
      let mut triplet = triplets[itriplet];
      let r = &mut triplet.0;
      let c = &mut triplet.1;
      if *r == drop_id || *c == drop_id {
        triplets.remove(itriplet);
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
  *galmat = SparseMatrix::from_triplets(ndofs_new, ndofs_new, triplets);
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
