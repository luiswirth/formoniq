use crate::{
  assemble::{self, GalMat},
  operators::{self, FeFunction},
};

use {
  common::sparse::{petsc_ghiep, petsc_saddle_point, SparseMatrix},
  exterior::ExteriorGrade,
  manifold::{geometry::metric::MeshEdgeLengths, topology::complex::Complex},
};

use itertools::Itertools;
use std::mem;

pub struct MixedGalmats {
  mass_sigma: GalMat,
  dif_sigma: GalMat,
  codif_u: GalMat,
  difdif_u: GalMat,
  mass_u: GalMat,
}
impl MixedGalmats {
  pub fn compute(topology: &Complex, geometry: &MeshEdgeLengths, grade: ExteriorGrade) -> Self {
    let (mass_sigma, dif_sigma, codif_u) = if grade > 0 {
      (
        assemble::assemble_galmat(topology, geometry, operators::HodgeMassElmat(grade - 1)),
        assemble::assemble_galmat(topology, geometry, operators::DifElmat(grade)),
        assemble::assemble_galmat(topology, geometry, operators::CodifElmat(grade)),
      )
    } else {
      (GalMat::default(), GalMat::default(), GalMat::default())
    };
    let difdif_u = assemble::assemble_galmat(topology, geometry, operators::CodifDifElmat(grade));
    let mass_u = assemble::assemble_galmat(topology, geometry, operators::HodgeMassElmat(grade));

    Self {
      mass_sigma,
      dif_sigma,
      codif_u,
      difdif_u,
      mass_u,
    }
  }

  pub fn sigma_len(&self) -> usize {
    self.mass_sigma.nrows()
  }
  pub fn u_len(&self) -> usize {
    self.mass_u.nrows()
  }

  pub fn mixed_hodge_laplacian(&self) -> SparseMatrix {
    let Self {
      mass_sigma,
      dif_sigma,
      codif_u,
      difdif_u,
      ..
    } = self;
    let codif_u = codif_u.clone();
    SparseMatrix::block(&[&[mass_sigma, &(-codif_u)], &[dif_sigma, difdif_u]])
  }
}

pub fn solve_hodge_laplace_source(
  topology: &Complex,
  geometry: &MeshEdgeLengths,
  grade: ExteriorGrade,
  source_data: FeFunction,
) -> (FeFunction, FeFunction, FeFunction) {
  let harmonics = solve_hodge_laplace_harmonics(topology, geometry, grade);

  let galmats = MixedGalmats::compute(topology, geometry, grade);

  let mass_u = galmats.mass_u.to_nalgebra_csr();
  let mass_harmonics = &mass_u * &harmonics;

  let sigma_len = galmats.sigma_len();
  let u_len = galmats.u_len();

  let mut galmat = galmats.mixed_hodge_laplacian();

  galmat.grow(mass_harmonics.ncols(), mass_harmonics.ncols());

  for (mut r, mut c) in (0..mass_harmonics.nrows()).cartesian_product(0..mass_harmonics.ncols()) {
    let v = mass_harmonics[(r, c)];
    r += sigma_len;
    c += sigma_len + u_len;
    galmat.push(r, c, v);
  }
  for (mut r, mut c) in (0..mass_harmonics.nrows()).cartesian_product(0..mass_harmonics.ncols()) {
    let v = mass_harmonics[(r, c)];
    // transpose
    mem::swap(&mut r, &mut c);
    r += sigma_len + u_len;
    c += sigma_len;
    galmat.push(r, c, v);
  }

  let galmat = galmat.to_nalgebra_csr();

  let galvec = mass_u * source_data.coeffs;
  #[allow(clippy::toplevel_ref_arg)]
  let galvec = na::stack![
    na::DVector::zeros(sigma_len);
    galvec;
    na::DVector::zeros(harmonics.ncols());
  ];

  let galsol = petsc_saddle_point(&galmat, &galvec);
  let sigma = FeFunction::new(grade - 1, galsol.view_range(..sigma_len, 0).into_owned());
  let u = FeFunction::new(
    grade,
    galsol
      .view_range(sigma_len..sigma_len + u_len, 0)
      .into_owned(),
  );
  let p = FeFunction::new(
    grade,
    galsol.view_range(sigma_len + u_len.., 0).into_owned(),
  );
  (sigma, u, p)
}

pub fn solve_hodge_laplace_harmonics(
  topology: &Complex,
  geometry: &MeshEdgeLengths,
  grade: ExteriorGrade,
) -> na::DMatrix<f64> {
  let homology_dim = topology.homology_dim(grade);

  if homology_dim == 0 {
    let nwhitneys = topology.nsimplicies(grade);
    return na::DMatrix::zeros(nwhitneys, 0);
  }

  let (eigenvals, harmonics) = solve_hodge_laplace_evp(topology, geometry, grade, homology_dim);
  assert!(eigenvals.iter().all(|&eigenval| eigenval <= 1e-12));
  harmonics
}

pub fn solve_hodge_laplace_evp(
  topology: &Complex,
  geometry: &MeshEdgeLengths,
  grade: ExteriorGrade,
  neigen_values: usize,
) -> (na::DVector<f64>, na::DMatrix<f64>) {
  let galmats = MixedGalmats::compute(topology, geometry, grade);

  let lhs = galmats.mixed_hodge_laplacian();

  let sigma_len = galmats.sigma_len();
  let u_len = galmats.u_len();
  let mut rhs = SparseMatrix::zeros(sigma_len + u_len, sigma_len + u_len);
  for &(mut r, mut c, v) in galmats.mass_u.triplets() {
    r += sigma_len;
    c += sigma_len;
    rhs.push(r, c, v);
  }

  petsc_ghiep(
    &lhs.to_nalgebra_csr(),
    &rhs.to_nalgebra_csr(),
    neigen_values,
  )
}
