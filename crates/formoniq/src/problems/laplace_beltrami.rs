//! Module for the Poisson Equation, the prototypical ellipitic PDE.

use crate::{
  assemble::GalVec,
  bc,
  whitney_complex::{BoundaryWhitneyComplex, WhitneyComplex},
};

use common::linalg::{
  nalgebra::{CsrMatrix, Vector},
  petsc::petsc_ghiep,
};
use ddf::cochain::Cochain;

/// Source problem of Laplace-Beltrami operator. Also known as Poisson Problem.
///
/// Essential boundary condition $"tr" u = g$ with `boundary_values` a
/// 0-cochain on the trace complex, imposed by affine lifting.
pub fn solve_laplace_beltrami_source(
  fes: WhitneyComplex,
  boundary: &BoundaryWhitneyComplex,
  source_galvec: GalVec,
  boundary_values: &Cochain,
) -> Cochain {
  let laplace = CsrMatrix::from(&fes.codif_dif(0));
  bc::solve_with_essential_bc(
    &fes.relative(),
    boundary,
    laplace,
    &source_galvec,
    boundary_values,
  )
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
