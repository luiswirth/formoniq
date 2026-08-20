use crate::{galerkin::GalerkinVector, hodge::HodgeBlocks, whitney_complex::HilbertComplex};

use {
  crate::linalg::{
    DirectInverse,
    eigen::{EigenError, sparse_shift_invert_eigen},
    faer::FaerLu,
  },
  derham::Cochain,
  multialgebra::ExteriorGrade,
};

use iterative::{BlockDiagonal, StopCriterion, krylov::minres};
use itertools::Itertools;
use simplicial::Dim;
use simplicial::linalg::{CooMatrix, CooMatrixExt, CsrMatrix, Matrix, Vector};
use std::mem;

/// The stable block-diagonal preconditioner for the mixed Hodge-Laplace
/// saddle point: the full $H Lambda(dif)$ inner product on each of the
/// $sigma$- and $u$-spaces (Arnold-Falk-Winther), the identity on the tiny
/// harmonic multiplier (the harmonics are mass-orthonormal). Each block is a
/// direct SPD solve.
///
/// `None` when a block is not positive definite, an indefinite mass on a
/// Lorentzian geometry, signaling the caller to fall back to a whole-system
/// indefinite factorization. This is the signature guard, read off the
/// factorization itself rather than a separate metric test.
pub fn mixed_block_preconditioner<C: HilbertComplex>(
  complex: &C,
  grade: ExteriorGrade,
  harmonic_len: usize,
) -> Option<BlockDiagonal<DirectInverse>> {
  let mut blocks = Vec::new();
  if grade > 0 {
    blocks.push(DirectInverse::try_new(complex.hdif_gram(grade - 1))?);
  }
  blocks.push(DirectInverse::try_new(complex.hdif_gram(grade))?);
  if harmonic_len > 0 {
    let mut identity = CooMatrix::zeros(harmonic_len, harmonic_len);
    for i in 0..harmonic_len {
      identity.push(i, i, 1.0);
    }
    blocks.push(DirectInverse::try_new(CsrMatrix::from(&identity))?);
  }
  Some(BlockDiagonal::new(blocks))
}

/// Assemble the augmented mixed Hodge-Laplace KKT system $(sigma, u, p)$ and its
/// right-hand side: the saddle point of the mixed formulation, bordered by the
/// harmonic constraint $chevron.l u, M h chevron.r = 0$ that fixes the solution
/// against the harmonic space. Returns the system matrix, the right-hand side,
/// and the $sigma$/$u$ block lengths.
pub fn assemble_mixed_kkt<C: HilbertComplex>(
  complex: &C,
  source_galvec: GalerkinVector,
  grade: ExteriorGrade,
  harmonics: &Matrix,
) -> (CsrMatrix, Vector, usize, usize) {
  let blocks = HodgeBlocks::compute(complex, grade);

  let mass_harmonics = &blocks.mass_u * harmonics;

  let sigma_len = blocks.n_sigma;
  let u_len = blocks.n_u;

  let mut galmat = blocks.mixed_hodge_laplacian();

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
  // relative one. The grading stops here, and has to: below, the three spaces
  // are stacked into one block system, an operator over their direct sum,
  // which no single grade names.
  let source = complex.inclusion(grade).transpose() * source_galvec.into_coeffs();

  #[allow(clippy::toplevel_ref_arg)]
  let rhs = na::stack![
    Vector::zeros(sigma_len);
    source;
    Vector::zeros(harmonics.ncols());
  ];

  // Symmetrize the saddle point by negating the $sigma$ equations. The mixed
  // form assembles the antisymmetric $sigma$-$u$ coupling ($-B^T$ above, $B$
  // below); negating the $sigma$ block-row turns it into the symmetric
  // $mat(-M, B^T; B, K)$. The $sigma$ right-hand side is zero, so the solution
  // is unchanged, and the symmetry is what lets MINRES solve it, while the
  // direct factorization is indifferent to it.
  let mut sign = CooMatrix::zeros(system_matrix.nrows(), system_matrix.nrows());
  for i in 0..system_matrix.nrows() {
    sign.push(i, i, if i < sigma_len { -1.0 } else { 1.0 });
  }
  let sign = CsrMatrix::from(&sign);
  (&sign * &system_matrix, &sign * &rhs, sigma_len, u_len)
}

/// The mixed Hodge-Laplace source problem $Delta u = f$ on any discrete Hilbert
/// complex: absolute (natural / Neumann) boundary conditions on the full
/// [`WhitneyComplex`], essential (homogeneous Dirichlet) on the
/// [`RelativeWhitneyComplex`], the same code either way.
///
/// The right-hand side `source_galvec` is assembled in the ambient
/// $cal(W) Lambda^k$. It is restricted to this complex's DOFs internally, and
/// the returned $(sigma, u, p)$ cochains are extended back to the ambient space,
/// so the caller is oblivious to the boundary condition. `p` is the harmonic
/// component of $u$, fixed to zero against the harmonic space $cal(H)^k$.
///
/// Fails only where [`solve_harmonics`] does: the harmonic basis
/// is an eigensolve.
///
/// [`WhitneyComplex`]: crate::whitney_complex::WhitneyComplex
/// [`RelativeWhitneyComplex`]: crate::whitney_complex::RelativeWhitneyComplex
pub fn solve_source<C: HilbertComplex>(
  complex: &C,
  source_galvec: GalerkinVector,
  grade: impl Into<ExteriorGrade>,
) -> Result<(Cochain, Cochain, Cochain), EigenError> {
  let grade = grade.into();
  let harmonics = solve_harmonics(complex, grade)?;
  let (system_matrix, rhs, sigma_len, u_len) =
    assemble_mixed_kkt(complex, source_galvec, grade, &harmonics);

  // The KKT system is symmetric indefinite. On a Riemannian geometry its
  // diagonal blocks are SPD, so a block-diagonal-preconditioned MINRES solves
  // it in an iteration count bounded independently of the mesh (the
  // Arnold-Falk-Winther norm equivalence), beating a direct factorization at
  // scale and avoiding its fill. On a Lorentzian geometry a block is indefinite,
  // the preconditioner cannot be built, and sparse LU carries the solve, so
  // the method stays total over signature. LU also catches the rare
  // non-convergence of the iterative path.
  let galsol = match mixed_block_preconditioner(complex, grade, harmonics.ncols()) {
    Some(precond) => {
      let (sol, report) = minres(&system_matrix, &precond, &rhs, StopCriterion::rtol(1e-10));
      if report.converged {
        sol
      } else {
        FaerLu::new(system_matrix).solve(&rhs)
      }
    }
    None => FaerLu::new(system_matrix).solve(&rhs),
  };

  // Extend the solution back to the ambient $cal(W) Lambda^k$ by zero on the
  // constrained boundary, $E u$, so callers see full cochains regardless of BC.
  let sigma_coeffs = galsol.view_range(..sigma_len, 0).into_owned();
  let u_coeffs = galsol
    .view_range(sigma_len..sigma_len + u_len, 0)
    .into_owned();
  let p_coeffs = galsol.view_range(sigma_len + u_len.., 0).into_owned();

  // At grade 0 the $sigma in Lambda^(-1)$ space is empty. There is nothing to
  // extend and no grade $-1$ to name it.
  let sigma = if grade > 0 {
    Cochain::new(grade - 1, complex.inclusion(grade - 1) * sigma_coeffs)
  } else {
    Cochain::new(Dim::ZERO, sigma_coeffs)
  };
  let u = Cochain::new(grade, complex.inclusion(grade) * u_coeffs);
  // `p` is the coefficient vector against the harmonic basis (length $b_k$), a
  // Lagrange multiplier — not a cochain in $u$-space, so it is not extended.
  let p = Cochain::new(grade, p_coeffs);
  Ok((sigma, u, p))
}

/// An $M_k$-orthonormal basis of the discrete harmonic space $cal(H)^k$, the
/// form the mixed saddle point of [`solve_source`] needs.
///
/// Computed by [`crate::harmonic::harmonics`], the $L^2$ projection of integral
/// cohomology generators, which is exact in $dif h = 0$ and takes its dimension
/// from topology rather than from an eigenvalue tolerance. On a geometry where
/// that projection is not well posed, an indefinite $L^2$ pairing, it falls
/// back to the shift-invert eigensolve of the Hodge-Laplace pencil near $0$,
/// keeping the method total over signature.
///
/// A caller wanting the harmonic forms of *specific* holes wants
/// [`crate::harmonic::Harmonics::integral`] instead; the orthonormalization
/// here mixes the classes.
pub fn solve_harmonics<C: HilbertComplex>(
  complex: &C,
  grade: impl Into<ExteriorGrade>,
) -> Result<Matrix, EigenError> {
  let grade = grade.into();
  // The dimension of the harmonic space is the Betti number $b_k$ (Hodge
  // theorem: $cal(H)^k tilde.equ H^k tilde.equ H_k$), an exact topological
  // invariant of the complex — not a number the caller has to know. Absolute
  // $b_k (K)$ on the full complex, relative $b_k (K, diff K)$ on the relative
  // one. The trait picks the right invariant.
  let homology_dim = complex.harmonic_dim(grade);
  if homology_dim == 0 {
    let nwhitneys = complex.ndofs(grade);
    return Ok(Matrix::zeros(nwhitneys, 0));
  }

  if let Some(harmonics) = crate::harmonic::harmonics(complex, grade) {
    return Ok(harmonics.orthonormal);
  }

  let (eigenvals, _, harmonics) = solve_evp(complex, grade, homology_dim)?;
  assert!(eigenvals.iter().all(|&eigenval| eigenval <= 1e-12));
  Ok(harmonics)
}

/// The `neigenvalues` eigenpairs of the mixed Hodge-Laplace pencil nearest
/// $0$, as $(lambda, sigma, u)$. Fewer, on a complex whose DOFs cannot support
/// that many.
pub fn solve_evp<C: HilbertComplex>(
  complex: &C,
  grade: impl Into<ExteriorGrade>,
  neigenvalues: usize,
) -> Result<(Vector, Matrix, Matrix), EigenError> {
  let grade = grade.into();
  let blocks = HodgeBlocks::compute(complex, grade);

  let lhs = blocks.mixed_hodge_laplacian();

  let (sigma_len, u_len) = (blocks.n_sigma, blocks.n_u);
  let mut rhs = CooMatrix::zeros(sigma_len + u_len, sigma_len + u_len);
  for (r, c, &v) in blocks.mass_u.triplet_iter() {
    rhs.push(sigma_len + r, sigma_len + c, v);
  }

  let (eigenvals, eigenvectors) =
    sparse_shift_invert_eigen(&(&lhs).into(), &(&rhs).into(), 0.0, neigenvalues)?;

  let eigen_sigmas = eigenvectors.rows(0, sigma_len).into_owned();
  let eigen_us = eigenvectors.rows(sigma_len, u_len).into_owned();
  Ok((eigenvals, eigen_sigmas, eigen_us))
}
