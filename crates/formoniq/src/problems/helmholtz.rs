use crate::{assemble, fe};

use lanczos::Hermitian;
use manifold::RiemannianComplex;

/// Eigenvalue problem of Laplace operator.
pub fn solve_helmholtz_homogeneous(
  mesh: &RiemannianComplex,
) -> (na::DVector<f64>, na::DMatrix<f64>) {
  let mut laplacian_glamat = assemble::assemble_galmat(mesh, fe::laplace_beltrami_elmat);
  let mut mass_galmat = assemble::assemble_galmat(mesh, fe::scalar_lumped_mass_elmat);

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

pub fn solve_hodge_laplace_evp() {}
