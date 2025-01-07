use crate::{assemble, fe};

use common::sparse::{petsc_ghiep, petsc_saddle_point, SparseMatrix};
use exterior::ExteriorRank;
use geometry::metric::manifold::MetricComplex;

pub fn solve_hodge_laplace_source(
  mesh: &MetricComplex,
  k: ExteriorRank,
  source_data: na::DVector<f64>,
) -> (na::DVector<f64>, na::DVector<f64>) {
  // TODO: handle harmonics (computed from EVP)
  //let harmonics = nas::CsrMatrix::zeros(0, 0);

  // TODO: handle k=0
  let mass_sigma = assemble::assemble_galmat(mesh, fe::whitney::HodgeMassElmat(k - 1));
  let dif_sigma = assemble::assemble_galmat(mesh, fe::whitney::DifElmat(k));
  let codif_u = assemble::assemble_galmat(mesh, fe::whitney::CodifElmat(k));
  let difdif_u = assemble::assemble_galmat(mesh, fe::whitney::CodifDifElmat(k));
  let mass_u = assemble::assemble_galmat(mesh, fe::whitney::HodgeMassElmat(k));

  let mut galmat = SparseMatrix::zeros(
    mass_sigma.nrows() + dif_sigma.nrows(),
    mass_sigma.ncols() + codif_u.ncols(),
  );
  for &(r, c, v) in mass_sigma.triplets() {
    galmat.push(r, c, v);
  }
  for &(r, mut c, mut v) in codif_u.triplets() {
    c += mass_sigma.ncols();
    v *= -1.0;
    galmat.push(r, c, v);
  }
  for &(mut r, c, v) in dif_sigma.triplets() {
    r += mass_sigma.nrows();
    galmat.push(r, c, v);
  }
  for &(mut r, mut c, v) in difdif_u.triplets() {
    r += mass_sigma.nrows();
    c += mass_sigma.ncols();
    galmat.push(r, c, v);
  }

  let galmat = galmat.to_nalgebra_csr();

  let mass_u = mass_u.to_nalgebra_csr();
  let galvec = mass_u * source_data;
  #[allow(clippy::toplevel_ref_arg)]
  let galvec = na::stack![
    na::DVector::zeros(mass_sigma.ncols());
    galvec
  ];

  let galsol = petsc_saddle_point(&galmat, &galvec);
  let sigma = galsol.view_range(..mass_sigma.nrows(), 0).into_owned();
  let u = galsol.view_range(mass_sigma.nrows().., 0).into_owned();
  (sigma, u)
}

pub fn solve_hodge_laplace_evp(
  mesh: &MetricComplex,
  k: ExteriorRank,
  neigen_values: usize,
) -> (na::DVector<f64>, na::DMatrix<f64>) {
  // TODO: handle k=0
  let mass_sigma = assemble::assemble_galmat(mesh, fe::whitney::HodgeMassElmat(k - 1));
  let dif_sigma = assemble::assemble_galmat(mesh, fe::whitney::DifElmat(k));
  let codif_u = assemble::assemble_galmat(mesh, fe::whitney::CodifElmat(k));
  let difdif_u = assemble::assemble_galmat(mesh, fe::whitney::CodifDifElmat(k));
  let mass_u = assemble::assemble_galmat(mesh, fe::whitney::HodgeMassElmat(k));

  let mut lhs = SparseMatrix::zeros(
    mass_sigma.nrows() + dif_sigma.nrows(),
    mass_sigma.ncols() + codif_u.ncols(),
  );
  for &(r, c, v) in mass_sigma.triplets() {
    lhs.push(r, c, v);
  }
  for &(r, mut c, mut v) in codif_u.triplets() {
    c += mass_sigma.ncols();
    v *= -1.0;
    lhs.push(r, c, v);
  }
  for &(mut r, c, v) in dif_sigma.triplets() {
    r += mass_sigma.nrows();
    lhs.push(r, c, v);
  }
  for &(mut r, mut c, v) in difdif_u.triplets() {
    r += mass_sigma.nrows();
    c += mass_sigma.ncols();
    lhs.push(r, c, v);
  }

  let mut rhs = SparseMatrix::zeros(
    mass_sigma.nrows() + mass_u.nrows(),
    mass_sigma.ncols() + mass_u.ncols(),
  );
  for &(mut r, mut c, v) in mass_u.triplets() {
    r += mass_sigma.nrows();
    c += mass_sigma.ncols();
    rhs.push(r, c, v);
  }

  petsc_ghiep(
    &lhs.to_nalgebra_csr(),
    &rhs.to_nalgebra_csr(),
    neigen_values,
  )
}
