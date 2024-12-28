use crate::{assemble, fe};

use common::sparse::SparseMatrix;
use exterior::ExteriorRank;
use manifold::RiemannianComplex;

#[allow(unused_variables)]
pub fn solve_hodge_laplace_evp(mesh: &RiemannianComplex, k: ExteriorRank) {
  let mass_galmat = assemble::assemble_galmat(mesh, fe::whitney::HodgeMassElmat(k));
  let id_dif_galmat = assemble::assemble_galmat(mesh, fe::whitney::IdDifElmat(k));
  let dif_id_galmat = assemble::assemble_galmat(mesh, fe::whitney::DifIdElmat(k));
  let dif_dif_galmat = assemble::assemble_galmat(mesh, fe::whitney::DifDifElmat(k));

  let lhs_matrix = SparseMatrix::zeros(
    mass_galmat.nrows() + dif_id_galmat.nrows(),
    mass_galmat.ncols() + id_dif_galmat.ncols(),
  );
  let rhs_matrix = SparseMatrix::zeros(
    mass_galmat.nrows() + dif_id_galmat.nrows(),
    mass_galmat.ncols() + id_dif_galmat.ncols(),
  );

  todo!()
}
