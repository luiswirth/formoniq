use crate::{
  assemble::{GalMat, GalVec},
  whitney_complex::HilbertComplex,
};

use {
  common::linalg::{eigen::sparse_shift_invert_eigen, faer::FaerLu},
  ddf::cochain::Cochain,
  exterior::ExteriorGrade,
};

use common::linalg::nalgebra::{CooMatrix, CooMatrixExt, CsrMatrix, Matrix, Vector};
use itertools::Itertools;
use std::mem;

/// The mixed Hodge-Laplace source problem $Delta u = f$ on any discrete Hilbert
/// complex: absolute (natural / Neumann) boundary conditions on the full
/// [`WhitneyComplex`], essential (homogeneous Dirichlet) on the
/// [`RelativeWhitneyComplex`] --- the same code either way.
///
/// The right-hand side `source_galvec` is assembled in the ambient
/// $cal(W) Lambda^k$; it is restricted to this complex's DOFs internally, and
/// the returned $(sigma, u, p)$ cochains are extended back to the ambient space,
/// so the caller is oblivious to the boundary condition. `p` is the harmonic
/// component of $u$, fixed to zero against the harmonic space $cal(H)^k$.
///
/// [`WhitneyComplex`]: crate::whitney_complex::WhitneyComplex
/// [`RelativeWhitneyComplex`]: crate::whitney_complex::RelativeWhitneyComplex
pub fn solve_hodge_laplace_source<C: HilbertComplex>(
  complex: &C,
  source_galvec: GalVec,
  grade: ExteriorGrade,
) -> (Cochain, Cochain, Cochain) {
  let harmonics = solve_hodge_laplace_harmonics(complex, grade);

  let galmats = MixedGalmats::compute(complex, grade);

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

  // Restrict the ambient right-hand side to this complex's DOFs, $E^T f$: the
  // identity on the full complex, a restriction to interior DOFs on the
  // relative one.
  let source = complex.inclusion(grade).transpose() * source_galvec;

  #[allow(clippy::toplevel_ref_arg)]
  let rhs = na::stack![
    Vector::zeros(sigma_len);
    source;
    Vector::zeros(harmonics.ncols());
  ];

  // The KKT system is symmetric indefinite, so Cholesky is out; sparse LU
  // solves it directly. Unsymmetric LU forfeits the ~2x a symmetric
  // $L D L^top$ would save on an indefinite system, nothing more.
  let galsol = FaerLu::new(system_matrix).solve(&rhs);

  // Extend the solution back to the ambient $cal(W) Lambda^k$ by zero on the
  // constrained boundary, $E u$, so callers see full cochains regardless of BC.
  let sigma_coeffs = galsol.view_range(..sigma_len, 0).into_owned();
  let u_coeffs = galsol
    .view_range(sigma_len..sigma_len + u_len, 0)
    .into_owned();
  let p_coeffs = galsol.view_range(sigma_len + u_len.., 0).into_owned();

  // At grade 0 the $sigma in Lambda^(-1)$ space is empty; there is nothing to
  // extend and no grade $-1$ to name it.
  let sigma = if grade > 0 {
    Cochain::new(grade - 1, complex.inclusion(grade - 1) * sigma_coeffs)
  } else {
    Cochain::new(0, sigma_coeffs)
  };
  let u = Cochain::new(grade, complex.inclusion(grade) * u_coeffs);
  // `p` is the coefficient vector against the harmonic basis (length $b_k$), a
  // Lagrange multiplier — not a cochain in $u$-space, so it is not extended.
  let p = Cochain::new(grade, p_coeffs);
  (sigma, u, p)
}

pub fn solve_hodge_laplace_harmonics<C: HilbertComplex>(
  complex: &C,
  grade: ExteriorGrade,
) -> Matrix {
  // The dimension of the harmonic space is the Betti number $b_k$ (Hodge
  // theorem: $cal(H)^k tilde.equ H^k tilde.equ H_k$), an exact topological
  // invariant of the complex — not a number the caller has to know. Absolute
  // $b_k (K)$ on the full complex, relative $b_k (K, diff K)$ on the relative
  // one; the trait picks the right invariant.
  let homology_dim = complex.harmonic_dim(grade);
  if homology_dim == 0 {
    let nwhitneys = complex.ndofs(grade);
    return Matrix::zeros(nwhitneys, 0);
  }

  let (eigenvals, _, harmonics) = solve_hodge_laplace_evp(complex, grade, homology_dim);
  assert!(eigenvals.iter().all(|&eigenval| eigenval <= 1e-12));
  harmonics
}

pub fn solve_hodge_laplace_evp<C: HilbertComplex>(
  complex: &C,
  grade: ExteriorGrade,
  neigenvalues: usize,
) -> (Vector, Matrix, Matrix) {
  let galmats = MixedGalmats::compute(complex, grade);

  let lhs = galmats.mixed_hodge_laplacian();

  let sigma_len = galmats.sigma_len();
  let u_len = galmats.u_len();
  let mut rhs = CooMatrix::zeros(sigma_len + u_len, sigma_len + u_len);
  for (mut r, mut c, &v) in galmats.mass_u.triplet_iter() {
    r += sigma_len;
    c += sigma_len;
    rhs.push(r, c, v);
  }

  let (eigenvals, eigenvectors) =
    sparse_shift_invert_eigen(&(&lhs).into(), &(&rhs).into(), 0.0, neigenvalues);

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
  pub fn compute<C: HilbertComplex>(complex: &C, grade: ExteriorGrade) -> Self {
    assert!(grade <= complex.dim());

    let mass_u = complex.mass(grade);
    let mass_u_csr = CsrMatrix::from(&mass_u);

    let (mass_sigma, dif_sigma, codif_u) = if grade > 0 {
      let mass_sigma = complex.mass(grade - 1);

      let exdif_sigma = complex.dif(grade - 1);

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

    let codifdif_u = if grade < complex.dim() {
      complex.codif_dif(grade)
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
