//! Module for the Poisson Equation, the prototypical ellipitic PDE.

use crate::{
  assemble::{self, GalVec},
  operators::DofCoeff,
  whitney_complex::WhitneyComplex,
};

use common::linalg::{
  faer::FaerCholesky,
  nalgebra::{CsrMatrix, Vector},
  petsc::petsc_ghiep,
};
use ddf::cochain::Cochain;
use manifold::topology::handle::KSimplexIdx;

/// Source problem of Laplace-Beltrami operator. Also known as Poisson Problem.
pub fn solve_laplace_beltrami_source<F>(
  fes: WhitneyComplex,
  mut source_galvec: GalVec,
  boundary_data: F,
) -> Cochain
where
  F: Fn(KSimplexIdx) -> DofCoeff,
{
  let mut laplace = fes.codif_dif(0);

  assemble::enforce_dirichlet_bc(
    fes.topology(),
    boundary_data,
    &mut laplace,
    &mut source_galvec,
  );

  let laplace = CsrMatrix::from(&laplace);
  let sol = FaerCholesky::new(laplace).solve(&source_galvec);
  Cochain::new(0, sol)
}

/// Eigenvalue problem of Laplace-Beltrami operator.
pub fn solve_laplace_beltrami_evp(
  fes: WhitneyComplex,
  neigen_values: usize,
) -> (Vector, Vec<Cochain>) {
  let laplace_galmat = fes.codif_dif(0);
  let mass_galmat = fes.mass(0);

  let (eigenvals, eigenvecs) = petsc_ghiep(
    &CsrMatrix::from(&laplace_galmat),
    &CsrMatrix::from(&mass_galmat),
    neigen_values,
  );

  let eigenvecs = eigenvecs
    .column_iter()
    .map(|c| Cochain::new(0, c.into_owned()))
    .collect();

  (eigenvals, eigenvecs)
}
