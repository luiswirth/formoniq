//! Module for the Poisson Equation, the prototypical ellipitic PDE.

use common::{sparse::petsc_eigensolve, util::FaerCholesky};
use manifold::RiemannianComplex;

use crate::{assemble, fe, fe::DofIdx};

/// Source problem of Laplace-Beltrami operator. Also known as Poisson Problem.
pub fn solve_laplace_beltrami_source<F>(
  mesh: &RiemannianComplex,
  source_data: na::DVector<f64>,
  boundary_data: F,
) -> na::DVector<f64>
where
  F: Fn(DofIdx) -> f64,
{
  let elmat = fe::LaplaceBeltramiElmat;
  let mut galmat = assemble::assemble_galmat(mesh, elmat);

  let elvec = fe::SourceElvec::new(source_data);
  let mut galvec = assemble::assemble_galvec(mesh, elvec);

  assemble::enforce_dirichlet_bc(mesh, boundary_data, &mut galmat, &mut galvec);

  let galmat = galmat.to_nalgebra_csr();
  FaerCholesky::new(galmat).solve(&galvec)
}

/// Eigenvalue problem of Laplace-Beltrami operator.
pub fn solve_laplace_beltrami_evp(
  mesh: &RiemannianComplex,
) -> (na::DVector<f64>, na::DMatrix<f64>) {
  let laplace_galmat = assemble::assemble_galmat(mesh, fe::LaplaceBeltramiElmat);
  let mass_galmat = assemble::assemble_galmat(mesh, fe::ScalarLumpedMassElmat);

  petsc_eigensolve(
    &laplace_galmat.to_nalgebra_csr(),
    &mass_galmat.to_nalgebra_csr(),
  )
}
