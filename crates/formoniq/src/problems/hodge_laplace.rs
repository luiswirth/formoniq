use std::mem;

use crate::{
  assemble::{self, GalMat},
  operators::{self, FeFunction},
};

use common::sparse::{petsc_ghiep, petsc_saddle_point, SparseMatrix};
use exterior::ExteriorGrade;
use geometry::metric::manifold::MetricComplex;
use itertools::Itertools;

pub fn solve_hodge_laplace_source(
  mesh: &MetricComplex,
  grade: ExteriorGrade,
  source_data: FeFunction,
) -> (FeFunction, FeFunction, FeFunction) {
  let harmonics = solve_hodge_laplace_harmonics(mesh, grade);

  let (mass_sigma, dif_sigma, codif_u) = if grade > 0 {
    (
      assemble::assemble_galmat(mesh, operators::HodgeMassElmat(grade - 1)),
      assemble::assemble_galmat(mesh, operators::DifElmat(grade)),
      assemble::assemble_galmat(mesh, operators::CodifElmat(grade)),
    )
  } else {
    (GalMat::default(), GalMat::default(), GalMat::default())
  };
  let difdif_u = assemble::assemble_galmat(mesh, operators::CodifDifElmat(grade));
  let mass_u = assemble::assemble_galmat(mesh, operators::HodgeMassElmat(grade));

  let mass_u = mass_u.to_nalgebra_csr();
  let mass_harmonics = &mass_u * &harmonics;

  let mut galmat = SparseMatrix::zeros(
    mass_sigma.nrows() + dif_sigma.nrows() + mass_harmonics.ncols(),
    mass_sigma.ncols() + codif_u.ncols() + mass_harmonics.ncols(),
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
  for (mut r, mut c) in (0..mass_harmonics.nrows()).zip(0..mass_harmonics.ncols()) {
    let v = mass_harmonics[(r, c)];
    r += mass_sigma.nrows();
    c += mass_sigma.ncols() + difdif_u.ncols();
    galmat.push(r, c, v);
  }
  for (mut r, mut c) in (0..mass_harmonics.nrows()).zip(0..mass_harmonics.ncols()) {
    let v = mass_harmonics[(r, c)];
    // transpose
    mem::swap(&mut r, &mut c);
    r += mass_sigma.nrows() + dif_sigma.nrows();
    c += mass_sigma.ncols();
    galmat.push(r, c, v);
  }

  let galmat = galmat.to_nalgebra_csr();

  let galvec = mass_u * source_data.coeffs;
  #[allow(clippy::toplevel_ref_arg)]
  let galvec = na::stack![
    na::DVector::zeros(mass_sigma.ncols());
    galvec;
    na::DVector::zeros(harmonics.ncols());
  ];

  let galsol = petsc_saddle_point(&galmat, &galvec);
  let sigma = FeFunction::new(
    grade - 1,
    galsol.view_range(..mass_sigma.ncols(), 0).into_owned(),
  );
  let u = FeFunction::new(
    grade,
    galsol
      .view_range(mass_sigma.ncols()..mass_sigma.ncols() + codif_u.ncols(), 0)
      .into_owned(),
  );
  let p = FeFunction::new(
    grade,
    galsol
      .view_range(mass_sigma.ncols() + codif_u.ncols().., 0)
      .into_owned(),
  );
  (sigma, u, p)
}

pub fn solve_hodge_laplace_harmonics(
  mesh: &MetricComplex,
  grade: ExteriorGrade,
) -> na::DMatrix<f64> {
  // TODO: improve this!
  // first of all find exactly all eigenvectors with eigenval 0
  // use simplical homology to determine number of harmonics
  let (eigenvals, eigenfuncs) = solve_hodge_laplace_evp(mesh, grade, 10);

  println!("{eigenvals}");

  if eigenvals[eigenvals.len() - 1] <= 1e-12 {
    panic!("might have missed a harmonic");
  }

  let harmonic_indices = eigenvals
    .iter()
    .enumerate()
    .filter(|&(_, &eigenval)| eigenval <= 1e-12)
    .map(|(i, _)| i)
    .collect_vec();

  let nwhitneys = mesh.topology().nsimplicies(grade);
  let mut harmonics = na::DMatrix::zeros(nwhitneys, harmonic_indices.len());
  for (icol, iharmonic) in harmonic_indices.into_iter().enumerate() {
    harmonics.set_column(icol, &eigenfuncs.column(iharmonic));
  }
  harmonics
}

pub fn solve_hodge_laplace_evp(
  mesh: &MetricComplex,
  grade: ExteriorGrade,
  neigen_values: usize,
) -> (na::DVector<f64>, na::DMatrix<f64>) {
  let (mass_sigma, dif_sigma, codif_u) = if grade > 0 {
    (
      assemble::assemble_galmat(mesh, operators::HodgeMassElmat(grade - 1)),
      assemble::assemble_galmat(mesh, operators::DifElmat(grade)),
      assemble::assemble_galmat(mesh, operators::CodifElmat(grade)),
    )
  } else {
    (GalMat::default(), GalMat::default(), GalMat::default())
  };
  let difdif_u = assemble::assemble_galmat(mesh, operators::CodifDifElmat(grade));
  let mass_u = assemble::assemble_galmat(mesh, operators::HodgeMassElmat(grade));

  let mut lhs = SparseMatrix::zeros(
    mass_sigma.nrows() + difdif_u.nrows(),
    mass_sigma.ncols() + difdif_u.ncols(),
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
