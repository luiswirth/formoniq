//! Elliptic Eigenvalue Problems

use lanczos::Hermitian;

use crate::{
  assemble,
  fe::{self, ElmatProvider},
  mesh::SimplicialManifold,
  space::FeSpace,
};

use std::rc::Rc;

// TODO: fix this
pub fn solve_homogeneous_evp(
  mesh: &Rc<SimplicialManifold>,
  operator_elmat: impl ElmatProvider,
) -> (na::DVector<f64>, na::DMatrix<f64>) {
  let space = FeSpace::new(Rc::clone(mesh));

  let mut operator_galmat = assemble::assemble_galmat(&space, operator_elmat);
  let mut mass_galmat = assemble::assemble_galmat(&space, fe::lumped_mass_elmat);

  assemble::drop_boundary_dofs_galmat(mesh, &mut operator_galmat);
  assemble::drop_boundary_dofs_galmat(mesh, &mut mass_galmat);

  // Convert generalized EVP into standard EVP
  // From $A u = lambda M u$ to $M^(-1) A u = lambda u$
  let mass_diagonal = mass_galmat.try_into_diagonal().unwrap();
  let mass_diagonal_inv = mass_diagonal.map(|x| x.recip());
  let system_matrix = operator_galmat.mul_left_by_diagonal(&mass_diagonal_inv);

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
