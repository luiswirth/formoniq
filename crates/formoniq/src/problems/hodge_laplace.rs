use crate::{assemble, fe};

use common::sparse::{petsc_ghiep, SparseMatrix};
use exterior::ExteriorRank;
use manifold::RiemannianComplex;

pub fn solve_hodge_laplace_evp(
  mesh: &RiemannianComplex,
  k: ExteriorRank,
) -> (na::DVector<f64>, na::DMatrix<f64>) {
  // TODO: should we accept negative ExteriorRanks?
  // TODO: return zero if elmat is zero for correspoding rank.
  let kkmass_galmat = assemble::assemble_galmat(mesh, fe::whitney::HodgeMassElmat(k - 1));
  let id_dif_galmat = assemble::assemble_galmat(mesh, fe::whitney::IdDifElmat(k));
  let dif_id_galmat = assemble::assemble_galmat(mesh, fe::whitney::DifIdElmat(k));
  let dif_dif_galmat = assemble::assemble_galmat(mesh, fe::whitney::DifDifElmat(k));
  let kmass_galmat = assemble::assemble_galmat(mesh, fe::whitney::HodgeMassElmat(k));

  let mut lhs = SparseMatrix::zeros(
    kkmass_galmat.nrows() + dif_id_galmat.nrows(),
    kkmass_galmat.ncols() + id_dif_galmat.ncols(),
  );
  for &(r, c, v) in kmass_galmat.triplets() {
    lhs.push(r, c, v);
  }
  for &(r, mut c, mut v) in id_dif_galmat.triplets() {
    c += kkmass_galmat.ncols();
    v *= -1.0;
    lhs.push(r, c, v);
  }
  for &(mut r, c, v) in dif_id_galmat.triplets() {
    r += kkmass_galmat.nrows();
    lhs.push(r, c, v);
  }
  for &(mut r, mut c, v) in dif_dif_galmat.triplets() {
    r += kkmass_galmat.nrows();
    c += kkmass_galmat.ncols();
    lhs.push(r, c, v);
  }

  let mut rhs = SparseMatrix::zeros(
    kkmass_galmat.nrows() + kmass_galmat.nrows(),
    kkmass_galmat.ncols() + kmass_galmat.ncols(),
  );
  for &(mut r, mut c, v) in kmass_galmat.triplets() {
    r += kkmass_galmat.nrows();
    c += kkmass_galmat.ncols();
    rhs.push(r, c, v);
  }

  petsc_ghiep(&lhs.to_nalgebra_csr(), &rhs.to_nalgebra_csr())
}
