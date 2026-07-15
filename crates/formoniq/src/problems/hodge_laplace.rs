use crate::{
  assemble::{GalMat, GalVec},
  whitney_complex::WhitneyComplex,
};

use {
  common::linalg::faer::{faer_ghiep, FaerLu},
  ddf::cochain::Cochain,
  exterior::ExteriorGrade,
};

use common::linalg::nalgebra::{CooMatrix, CooMatrixExt, CsrMatrix, Matrix, Vector};
use itertools::Itertools;
use std::mem;

pub fn solve_hodge_laplace_source(
  whitney: WhitneyComplex,
  source_galvec: GalVec,
  grade: ExteriorGrade,
) -> (Cochain, Cochain, Cochain) {
  let harmonics = solve_hodge_laplace_harmonics(whitney, grade);

  let galmats = MixedGalmats::compute(whitney, grade);

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

  let system_matrix = CsrMatrix::from(&galmat);

  #[allow(clippy::toplevel_ref_arg)]
  let rhs = na::stack![
    Vector::zeros(sigma_len);
    source_galvec;
    Vector::zeros(harmonics.ncols());
  ];

  // The KKT system is symmetric indefinite, so Cholesky is out; sparse LU
  // solves it directly. Unsymmetric LU forfeits the ~2x a symmetric
  // $L D L^top$ would save on an indefinite system, nothing more.
  let galsol = FaerLu::new(system_matrix).solve(&rhs);
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

pub fn solve_hodge_laplace_harmonics(whitney: WhitneyComplex, grade: ExteriorGrade) -> Matrix {
  // The dimension of the harmonic space is the Betti number $b_k$ (Hodge
  // theorem: $cal(H)^k tilde.equ H^k tilde.equ H_k$), an exact topological
  // invariant of the complex — not a number the caller has to know.
  let homology_dim = whitney.topology().betti_number(grade);
  if homology_dim == 0 {
    let nwhitneys = whitney.ndofs(grade);
    return Matrix::zeros(nwhitneys, 0);
  }

  let (eigenvals, _, harmonics) = solve_hodge_laplace_evp(whitney, grade, homology_dim);
  assert!(eigenvals.iter().all(|&eigenval| eigenval <= 1e-12));
  harmonics
}

pub fn solve_hodge_laplace_evp(
  whitney: WhitneyComplex,
  grade: ExteriorGrade,
  neigenvalues: usize,
) -> (Vector, Matrix, Matrix) {
  let galmats = MixedGalmats::compute(whitney, grade);

  let lhs = galmats.mixed_hodge_laplacian();

  let sigma_len = galmats.sigma_len();
  let u_len = galmats.u_len();
  let mut rhs = CooMatrix::zeros(sigma_len + u_len, sigma_len + u_len);
  for (mut r, mut c, &v) in galmats.mass_u.triplet_iter() {
    r += sigma_len;
    c += sigma_len;
    rhs.push(r, c, v);
  }

  let (eigenvals, eigenvectors) = faer_ghiep(&(&lhs).into(), &(&rhs).into(), neigenvalues);

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
  pub fn compute(whitney: WhitneyComplex, grade: ExteriorGrade) -> Self {
    assert!(grade <= whitney.dim());

    let mass_u = whitney.mass(grade);
    let mass_u_csr = CsrMatrix::from(&mass_u);

    let (mass_sigma, dif_sigma, codif_u) = if grade > 0 {
      let mass_sigma = whitney.mass(grade - 1);

      let exdif_sigma = whitney.dif(grade - 1);

      let dif_sigma = &mass_u_csr * &exdif_sigma;
      let dif_sigma = CooMatrix::from(&dif_sigma);

      let codif_u = &exdif_sigma.transpose() * &mass_u_csr;
      let codif_u = CooMatrix::from(&codif_u);

      (mass_sigma, dif_sigma, codif_u)
    } else {
      // At grade 0 the $sigma$ space is empty, but the off-diagonal blocks still
      // have to align with the $u$ block: $dif_sigma$ maps the empty $sigma$ into
      // $u$-space ($u_"len" times 0$) and $codif_u$ maps $u$ into it
      // ($0 times u_"len"$). Shaping them $0 times 0$ instead breaks block assembly.
      let u_len = mass_u.nrows();
      (
        GalMat::new(0, 0),
        GalMat::new(u_len, 0),
        GalMat::new(0, u_len),
      )
    };

    let codifdif_u = if grade < whitney.dim() {
      whitney.codif_dif(grade)
    } else {
      // At top grade $dif u = 0$, so the $codif dif$ block vanishes — but it still
      // occupies the $u times u$ diagonal slot and must be shaped $u_"len"^2$, not
      // $0 times 0$, or block assembly misaligns.
      GalMat::new(mass_u.nrows(), mass_u.nrows())
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
