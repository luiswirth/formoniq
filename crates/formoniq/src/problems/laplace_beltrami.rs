//! Module for the Poisson Equation, the prototypical ellipitic PDE.

use common::util::FaerCholesky;
use lanczos::Hermitian;
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

  let galmat = galmat.to_nalgebra_csc();
  FaerCholesky::new(galmat).solve(&galvec)
}

/// Eigenvalue problem of Laplace-Beltrami operator.
pub fn solve_laplace_beltrami_evp(
  mesh: &RiemannianComplex,
) -> (na::DVector<f64>, na::DMatrix<f64>) {
  let mut laplacian_glamat = assemble::assemble_galmat(mesh, fe::LaplaceBeltramiElmat);
  let mut mass_galmat = assemble::assemble_galmat(mesh, fe::ScalarLumpedMassElmat);

  assemble::drop_boundary_dofs_galmat(mesh, &mut laplacian_glamat);
  assemble::drop_boundary_dofs_galmat(mesh, &mut mass_galmat);

  // Convert generalized EVP into standard EVP
  // From $A u = lambda M u$ to $M^(-1) A u = lambda u$
  let mass_diagonal = mass_galmat.try_into_diagonal().unwrap();
  let mass_diagonal_inv = mass_diagonal.map(|x| x.recip());
  let system_matrix = laplacian_glamat.mul_left_by_diagonal(&mass_diagonal_inv);

  // Solve standard EVP with Lanczos algorithm
  let system_matrix = system_matrix.to_nalgebra_csc();
  let eigen = system_matrix.eigsh(10, lanczos::Order::Smallest);
  let eigenvals = eigen.eigenvalues;
  let mut eigenfuncs = eigen.eigenvectors;

  eigenfuncs
    .column_iter_mut()
    .for_each(|mut c| c.component_mul_assign(&mass_diagonal_inv));

  assemble::reintroduce_zeroed_boundary_dofs_galsols(mesh, &mut eigenfuncs);

  (eigenvals, eigenfuncs)
}
