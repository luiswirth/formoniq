//! Module for the Poisson Equation, the prototypical ellipitic PDE.

use crate::{
  assemble,
  fe::{self, DofCoeff, FeFunction},
};

use common::{sparse::petsc_ghiep, util::FaerCholesky};
use geometry::metric::manifold::MetricComplex;
use topology::complex::handle::KSimplexIdx;

/// Source problem of Laplace-Beltrami operator. Also known as Poisson Problem.
pub fn solve_laplace_beltrami_source<F>(
  mesh: &MetricComplex,
  source_data: FeFunction,
  boundary_data: F,
) -> FeFunction
where
  F: Fn(KSimplexIdx) -> DofCoeff,
{
  let mut laplace = assemble::assemble_galmat(mesh, fe::LaplaceBeltramiElmat);

  let mass = assemble::assemble_galmat(mesh, fe::ScalarMassElmat).to_nalgebra_csr();
  let mut source = mass * source_data;

  assemble::enforce_dirichlet_bc(mesh.topology(), boundary_data, &mut laplace, &mut source);

  let laplace = laplace.to_nalgebra_csr();
  FaerCholesky::new(laplace).solve(&source)
}

/// Eigenvalue problem of Laplace-Beltrami operator.
pub fn solve_laplace_beltrami_evp(
  mesh: &MetricComplex,
  neigen_values: usize,
) -> (na::DVector<f64>, na::DMatrix<f64>) {
  let laplace_galmat = assemble::assemble_galmat(mesh, fe::LaplaceBeltramiElmat);
  let mass_galmat = assemble::assemble_galmat(mesh, fe::ScalarLumpedMassElmat);

  petsc_ghiep(
    &laplace_galmat.to_nalgebra_csr(),
    &mass_galmat.to_nalgebra_csr(),
    neigen_values,
  )
}
