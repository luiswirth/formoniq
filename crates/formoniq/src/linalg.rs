//! The solve machinery: a faer bridge for sparse LU/Cholesky, shift-invert
//! eigensolving, and the sparse bilinear-form helpers assembly needs on top of
//! `simplicial`'s matrix types.
//!
//! `formoniq` is the only crate in the workspace that actually solves a linear
//! system, so this is where faer lives -- not a shared base crate every leaf
//! would otherwise compile it for free.

use simplicial::linalg::{CsrMatrix, Vector};

use crate::linalg::faer::{FaerCholesky, FaerLu};

pub mod eigen;
pub mod faer;

pub fn bilinear_form_sparse(mat: &CsrMatrix, u: &Vector, v: &Vector) -> f64 {
  ((mat.transpose() * u).transpose() * v).x
}
pub fn quadratic_form_sparse(mat: &CsrMatrix, u: &Vector) -> f64 {
  bilinear_form_sparse(mat, u, u)
}

/// The chosen factorization behind a [`DirectInverse`]: Cholesky by default, LU
/// as the verified fallback.
enum Factorization {
  Cholesky(Box<FaerCholesky>),
  Lu(Box<FaerLu>),
}

/// A direct SPD factorization presented as an [`iterative::ApproxInverse`]: the
/// exact inverse $B = A^(-1)$, the perfect approximate inverse.
///
/// The bridge that lets a direct factorization serve as a *block* of a
/// preconditioner --- an exact inner solve on one space of a saddle point ---
/// so a block-diagonal preconditioner over these makes MINRES converge in an
/// iteration count independent of the mesh. Self-adjoint by construction: the
/// inverse of an SPD matrix is SPD, whichever factorization computes it.
///
/// Cholesky is the natural choice for an SPD system and the default here, but
/// faer's sparse Cholesky (0.24) silently returns a grossly inaccurate
/// factorization on some large SPD systems (a 3D grade-0 Hodge-Laplace at ~10^5
/// DOFs solves to a relative residual of only ~10^-4, while faer's *LU* on the
/// same matrix is exact to machine precision). A wrong answer from a direct
/// solver is the worst failure mode, so the factorization is *verified* at
/// construction against a probe right-hand side and falls back to LU when the
/// Cholesky solve does not reproduce it. The guard costs one extra triangular
/// solve, negligible next to the factorization.
pub struct DirectInverse {
  factorization: Factorization,
  dim: usize,
}

/// A Cholesky solve is trusted when it reproduces a probe right-hand side to
/// near machine precision. The faer accuracy failure is right-hand-side
/// independent (the factor itself is corrupt), so one probe settles it.
const PROBE_RTOL: f64 = 1e-8;

impl DirectInverse {
  /// Factor an SPD block, `None` if it is not positive definite (so a caller
  /// building a block preconditioner can fall back to a whole-system indefinite
  /// solve on a Lorentzian geometry).
  ///
  /// Prefers Cholesky; verifies it and falls back to LU if faer's sparse
  /// Cholesky returned an inaccurate factor (see the type docs). `None` only when
  /// the matrix is genuinely not positive definite -- Cholesky failing the PD
  /// check, not merely the accuracy probe.
  pub fn try_new(a: CsrMatrix) -> Option<Self> {
    let dim = a.nrows();
    // `None` here means genuinely indefinite: preserve that contract.
    let chol = FaerCholesky::try_new(a.clone())?;

    let factorization = if cholesky_is_accurate(&a, &chol) {
      Factorization::Cholesky(Box::new(chol))
    } else {
      // The matrix is PD (Cholesky's PD check passed); LU is unconditionally
      // applicable and, empirically, accurate where the Cholesky factor is not.
      Factorization::Lu(Box::new(FaerLu::new(a)))
    };
    Some(Self { factorization, dim })
  }
}

/// Whether `chol` actually solves `a x = b`: does the factorization reproduce a
/// deterministic probe right-hand side to [`PROBE_RTOL`]?
fn cholesky_is_accurate(a: &CsrMatrix, chol: &FaerCholesky) -> bool {
  let n = a.nrows();
  if n == 0 {
    return true;
  }
  let b = Vector::from_fn(n, |i, _| ((i as f64 + 1.0) * 0.5).sin());
  let x = chol.solve(&b);
  (a * x - &b).norm() <= PROBE_RTOL * b.norm()
}

impl iterative::ApproxInverse for DirectInverse {
  fn dim(&self) -> usize {
    self.dim
  }
  fn apply(&self, r: &Vector) -> Vector {
    match &self.factorization {
      Factorization::Cholesky(chol) => chol.solve(r),
      Factorization::Lu(lu) => lu.solve(r),
    }
  }
}
impl iterative::SelfAdjoint for DirectInverse {}
