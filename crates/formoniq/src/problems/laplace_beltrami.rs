//! Module for the Poisson Equation, the prototypical ellipitic PDE.

use crate::{
  assemble,
  operators::{self, DofCoeff},
};

use common::{sparse::petsc_ghiep, util::FaerCholesky};
use manifold::{
  geometry::metric::MeshEdgeLengths,
  topology::complex::{attribute::Cochain, handle::KSimplexIdx, Complex},
};

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

  let mass =
    assemble::assemble_galmat(topology, geometry, operators::ScalarMassElmat).to_nalgebra_csr();
  let mut source = mass * source_data.coeffs;

  assemble::enforce_dirichlet_bc(topology, boundary_data, &mut laplace, &mut source);

  let laplace = laplace.to_nalgebra_csr();
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
  let mass_galmat = assemble::assemble_galmat(topology, geometry, operators::ScalarLumpedMassElmat);

  let (eigenvals, eigenvecs) = petsc_ghiep(
    &laplace_galmat.to_nalgebra_csr(),
    &mass_galmat.to_nalgebra_csr(),
    neigen_values,
  );

  let eigenvecs = eigenvecs
    .column_iter()
    .map(|c| Cochain::new(0, c.into_owned()))
    .collect();

  (eigenvals, eigenvecs)
}
