//! Module for the Poisson Equation, the prototypical ellipitic PDE.

use crate::{
  assemble,
  operators::{self, DofCoeff},
};

use common::linalg::{faer::FaerCholesky, petsc::petsc_ghiep};
use manifold::{
  geometry::metric::MeshEdgeLengths,
  topology::complex::{handle::KSimplexIdx, Complex},
};
use whitney::cochain::Cochain;

/// Source problem of Laplace-Beltrami operator. Also known as Poisson Problem.
pub fn solve_laplace_beltrami_source<F>(
  topology: &Complex,
  geometry: &MeshEdgeLengths,
  source_data: Cochain,
  boundary_data: F,
) -> Cochain
where
  F: Fn(KSimplexIdx) -> DofCoeff,
{
  let mut laplace = assemble::assemble_galmat(topology, geometry, operators::LaplaceBeltramiElmat);

  let mass = assemble::assemble_galmat(topology, geometry, operators::ScalarMassElmat);
  let mass = nas::CsrMatrix::from(&mass);
  let mut source = mass * source_data.coeffs;

  assemble::enforce_dirichlet_bc(topology, boundary_data, &mut laplace, &mut source);

  let laplace = nas::CsrMatrix::from(&laplace);
  let sol = FaerCholesky::new(laplace).solve(&source);
  Cochain::new(0, sol)
}

/// Eigenvalue problem of Laplace-Beltrami operator.
pub fn solve_laplace_beltrami_evp(
  topology: &Complex,
  geometry: &MeshEdgeLengths,
  neigen_values: usize,
) -> (na::DVector<f64>, Vec<Cochain>) {
  let laplace_galmat =
    assemble::assemble_galmat(topology, geometry, operators::LaplaceBeltramiElmat);
  let mass_galmat = assemble::assemble_galmat(topology, geometry, operators::ScalarMassElmat);

  let (eigenvals, eigenvecs) = petsc_ghiep(
    &nas::CsrMatrix::from(&laplace_galmat),
    &nas::CsrMatrix::from(&mass_galmat),
    neigen_values,
  );

  let eigenvecs = eigenvecs
    .column_iter()
    .map(|c| Cochain::new(0, c.into_owned()))
    .collect();

  (eigenvals, eigenvecs)
}
