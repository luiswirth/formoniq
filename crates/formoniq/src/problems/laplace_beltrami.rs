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
  whitney: WhitneyComplex,
  boundary: &BoundaryWhitneyComplex,
  source_galvec: GalVec,
  boundary_values: &Cochain,
) -> Cochain {
  let laplace = CsrMatrix::from(&whitney.codif_dif(0));
  bc::solve_with_essential_bc(
    &whitney.relative(),
    boundary,
    laplace,
    &source_galvec,
    boundary_values,
  )
}

/// Eigenvalue problem of Laplace-Beltrami operator.
pub fn solve_laplace_beltrami_evp(
  whitney: WhitneyComplex,
  neigenvalues: usize,
) -> (Vector, Vec<Cochain>) {
  let laplace_galmat = whitney.codif_dif(0);
  let mass_galmat = whitney.mass(0);

  let (eigenvals, eigenvecs) = petsc_ghiep(
    &CsrMatrix::from(&laplace_galmat),
    &CsrMatrix::from(&mass_galmat),
    neigenvalues,
  );

  let eigenvecs = eigenvecs
    .column_iter()
    .map(|c| Cochain::new(0, c.into_owned()))
    .collect();

  (eigenvals, eigenvecs)
}
