//! Module for the Poisson Equation, the prototypical ellipitic PDE.

use crate::{
  assemble::{self, GalVec},
  operators::{self, DofCoeff},
};

use common::linalg::{
  faer::FaerCholesky,
  nalgebra::{CsrMatrix, Vector},
  petsc::petsc_ghiep,
};
use ddf::cochain::Cochain;
use manifold::{
  geometry::metric::mesh::MeshLengths,
  topology::{complex::Complex, handle::KSimplexIdx},
};

/// Source problem of Laplace-Beltrami operator. Also known as Poisson Problem.
pub fn solve_laplace_beltrami_source<F>(
  topology: &Complex,
  geometry: &MeshLengths,
  mut source_galvec: GalVec,
  boundary_data: F,
) -> Cochain
where
  F: Fn(KSimplexIdx) -> DofCoeff,
{
  let dim = topology.dim();
  let mut laplace = assemble::assemble_galmat(
    topology,
    geometry,
    operators::LaplaceBeltramiElmat::new(dim),
  );

  assemble::enforce_dirichlet_bc(topology, boundary_data, &mut laplace, &mut source_galvec);

  let laplace = CsrMatrix::from(&laplace);
  let sol = FaerCholesky::new(laplace).solve(&source_galvec);
  Cochain::new(0, sol)
}

/// Eigenvalue problem of Laplace-Beltrami operator.
pub fn solve_laplace_beltrami_evp(
  topology: &Complex,
  geometry: &MeshLengths,
  neigen_values: usize,
) -> (Vector, Vec<Cochain>) {
  let dim = topology.dim();
  let laplace_galmat = assemble::assemble_galmat(
    topology,
    geometry,
    operators::LaplaceBeltramiElmat::new(dim),
  );
  let mass_galmat = assemble::assemble_galmat(topology, geometry, operators::ScalarMassElmat);

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
