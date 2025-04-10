use crate::{
  assemble::{assemble_galmat, GalMat},
  operators::HodgeMassElmat,
};

use {
  common::linalg::petsc::{petsc_ghiep, petsc_saddle_point},
  exterior::ExteriorGrade,
  manifold::{geometry::metric::mesh::MeshLengths, topology::complex::Complex},
  whitney::{cochain::Cochain, ManifoldComplexExt},
};

use common::linalg::nalgebra::{CooMatrix, CooMatrixExt, CsrMatrix, Matrix, Vector};
use itertools::Itertools;
use std::mem;

pub fn hodge_decomposition(
  topology: &Complex,
  geometry: &MeshLengths,
  cochain: Cochain,
) -> (Cochain, Cochain, Cochain) {
  let grade = cochain.dim();
  let (exact_potential, coexact, harmonic) =
    solve_hodge_laplace_source(topology, geometry, cochain);
  let dif = CsrMatrix::from(&topology.exterior_derivative_operator(grade - 1));
  let exact = Cochain::new(grade, dif * exact_potential.coeffs.clone());
  (exact, coexact, harmonic)
}

pub fn solve_hodge_laplace_source(
  topology: &Complex,
  geometry: &MeshLengths,
  source_data: Cochain,
) -> (Cochain, Cochain, Cochain) {
  let grade = source_data.dim();

  let harmonics = solve_hodge_laplace_harmonics(topology, geometry, grade);

  let galmats = MixedGalmats::compute(topology, geometry, grade);

  let mass_u = CsrMatrix::from(&galmats.mass_u);
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

  let galmat = CsrMatrix::from(&galmat);

  let galvec = mass_u * source_data.coeffs;
  #[allow(clippy::toplevel_ref_arg)]
  let galvec = na::stack![
    Vector::zeros(sigma_len);
    galvec;
    Vector::zeros(harmonics.ncols());
  ];

  let galsol = petsc_saddle_point(&galmat, &galvec);
  let sigma = Cochain::new(grade - 1, galsol.view_range(..sigma_len, 0).into_owned());
  let u = Cochain::new(
    grade,
    galsol
      .view_range(sigma_len..sigma_len + u_len, 0)
      .into_owned(),
  );
  let p = Cochain::new(
    grade,
    galsol.view_range(sigma_len + u_len.., 0).into_owned(),
  );
  (sigma, u, p)
}

pub fn solve_hodge_laplace_harmonics(
  topology: &Complex,
  geometry: &MeshLengths,
  grade: ExteriorGrade,
) -> Matrix {
  // TODO!!!
  //let homology_dim = topology.homology_dim(grade);
  let homology_dim = 0;

  if homology_dim == 0 {
    let nwhitneys = topology.nsimplicies(grade);
    return Matrix::zeros(nwhitneys, 0);
  }

  let (eigenvals, _, harmonics) = solve_hodge_laplace_evp(topology, geometry, grade, homology_dim);
  assert!(eigenvals.iter().all(|&eigenval| eigenval <= 1e-12));
  harmonics
}

pub fn solve_hodge_laplace_evp(
  topology: &Complex,
  geometry: &MeshLengths,
  grade: ExteriorGrade,
  neigen_values: usize,
) -> (Vector, Matrix, Matrix) {
  let galmats = MixedGalmats::compute(topology, geometry, grade);

  let lhs = galmats.mixed_hodge_laplacian();

  let sigma_len = galmats.sigma_len();
  let u_len = galmats.u_len();
  let mut rhs = CooMatrix::zeros(sigma_len + u_len, sigma_len + u_len);
  for (mut r, mut c, &v) in galmats.mass_u.triplet_iter() {
    r += sigma_len;
    c += sigma_len;
    rhs.push(r, c, v);
  }

  let (eigenvals, eigenvectors) = petsc_ghiep(&(&lhs).into(), &(&rhs).into(), neigen_values);

  let eigen_sigmas = eigenvectors.rows(0, sigma_len).into_owned();
  let eigen_us = eigenvectors.rows(sigma_len, u_len).into_owned();
  (eigenvals, eigen_sigmas, eigen_us)
}

pub struct MixedGalmats {
  mass_sigma: GalMat,
  dif_sigma: GalMat,
  codif_u: GalMat,
  codifdif_u: GalMat,
  mass_u: GalMat,
}
impl MixedGalmats {
  pub fn compute(topology: &Complex, geometry: &MeshLengths, grade: ExteriorGrade) -> Self {
    assert!(grade <= topology.dim());

    let mass_u = assemble_galmat(topology, geometry, HodgeMassElmat(grade));
    let mass_u_csr = CsrMatrix::from(&mass_u);

    let (mass_sigma, dif_sigma, codif_u) = if grade > 0 {
      let mass_sigma = assemble_galmat(topology, geometry, HodgeMassElmat(grade - 1));

      let exdif_sigma = topology.exterior_derivative_operator(grade - 1);
      let exdif_sigma = CsrMatrix::from(&exdif_sigma);

      let dif_sigma = &mass_u_csr * &exdif_sigma;
      let dif_sigma = CooMatrix::from(&dif_sigma);

      let codif_u = &exdif_sigma.transpose() * &mass_u_csr;
      let codif_u = CooMatrix::from(&codif_u);

      (mass_sigma, dif_sigma, codif_u)
    } else {
      (GalMat::new(0, 0), GalMat::new(0, 0), GalMat::new(0, 0))
    };

    let codifdif_u = if grade < topology.dim() {
      let mass_plus = assemble_galmat(topology, geometry, HodgeMassElmat(grade + 1));
      let mass_plus = CsrMatrix::from(&mass_plus);
      let exdif_u = topology.exterior_derivative_operator(grade);
      let exdif_u = CsrMatrix::from(&exdif_u);
      let codifdif_u = exdif_u.transpose() * mass_plus * exdif_u;
      CooMatrix::from(&codifdif_u)
    } else {
      GalMat::new(0, 0)
    };

    Self {
      mass_sigma,
      dif_sigma,
      codif_u,
      codifdif_u,
      mass_u,
    }
  }

  pub fn sigma_len(&self) -> usize {
    self.mass_sigma.nrows()
  }
  pub fn u_len(&self) -> usize {
    self.mass_u.nrows()
  }

  pub fn mixed_hodge_laplacian(&self) -> CooMatrix {
    let Self {
      mass_sigma,
      dif_sigma,
      codif_u,
      codifdif_u,
      ..
    } = self;
    let codif_u = codif_u.clone();
    CooMatrix::block(&[&[mass_sigma, &(codif_u.neg())], &[dif_sigma, codifdif_u]])
  }
}
