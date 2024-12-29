use crate::{assemble, fe};

use common::sparse::{petsc_write_binary, SparseMatrix};
use exterior::ExteriorRank;
use manifold::RiemannianComplex;

#[allow(unused_variables)]
pub fn solve_hodge_laplace_evp(mesh: &RiemannianComplex, k: ExteriorRank) {
  let (lhs, rhs) = if k == 0 {
    let dif_dif_galmat = assemble::assemble_galmat(mesh, fe::whitney::DifDifElmat(k));
    let kmass_galmat = assemble::assemble_galmat(mesh, fe::whitney::HodgeMassElmat(k));

    (dif_dif_galmat, kmass_galmat)
  } else {
    let kkmass_galmat = assemble::assemble_galmat(mesh, fe::whitney::HodgeMassElmat(k - 1));
    let kmass_galmat = assemble::assemble_galmat(mesh, fe::whitney::HodgeMassElmat(k));
    let id_dif_galmat = assemble::assemble_galmat(mesh, fe::whitney::IdDifElmat(k));
    let dif_id_galmat = assemble::assemble_galmat(mesh, fe::whitney::DifIdElmat(k));
    let dif_dif_galmat = assemble::assemble_galmat(mesh, fe::whitney::DifDifElmat(k));

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

    (lhs, rhs)
  };

  petsc_write_binary(
    &lhs.to_nalgebra_csr(),
    "/home/luis/thesis/formoniq/out/lhs.bin",
  )
  .unwrap();
  petsc_write_binary(
    &rhs.to_nalgebra_csr(),
    "/home/luis/thesis/formoniq/out/rhs.bin",
  )
  .unwrap();
}
