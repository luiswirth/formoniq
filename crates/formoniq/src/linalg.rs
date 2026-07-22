//! The solve machinery: a faer bridge for sparse LU/Cholesky, shift-invert
//! eigensolving, and the sparse bilinear-form helpers assembly needs on top of
//! `simplicial`'s matrix types.
//!
//! `formoniq` is the only crate in the workspace that actually solves a linear
//! system, so this is where faer lives -- not a shared base crate every leaf
//! would otherwise compile it for free.

use simplicial::linalg::{CsrMatrix, Vector};

use crate::linalg::faer::FaerCholesky;

pub mod eigen;
pub mod faer;

pub fn bilinear_form_sparse(mat: &CsrMatrix, u: &Vector, v: &Vector) -> f64 {
  ((mat.transpose() * u).transpose() * v).x
}
pub fn quadratic_form_sparse(mat: &CsrMatrix, u: &Vector) -> f64 {
  bilinear_form_sparse(mat, u, u)
}

/// A direct SPD factorization presented as an [`iterative::ApproxInverse`]: the
/// exact inverse $B = A^(-1)$, the perfect approximate inverse.
///
/// The bridge that lets a Cholesky factorization serve as a *block* of a
/// preconditioner --- an exact inner solve on one space of a saddle point ---
/// so a block-diagonal preconditioner over these makes MINRES converge in an
/// iteration count independent of the mesh. Self-adjoint by construction (the
/// inverse of an SPD matrix is SPD).
pub struct DirectInverse {
  chol: FaerCholesky,
  dim: usize,
}

impl DirectInverse {
  /// Factor an SPD block, `None` if it is not positive definite (so a caller
  /// building a block preconditioner can fall back to a whole-system indefinite
  /// solve on a Lorentzian geometry).
  pub fn try_new(a: CsrMatrix) -> Option<Self> {
    let dim = a.nrows();
    FaerCholesky::try_new(a).map(|chol| Self { chol, dim })
  }
}

impl iterative::ApproxInverse for DirectInverse {
  fn dim(&self) -> usize {
    self.dim
  }
  fn apply(&self, r: &Vector) -> Vector {
    self.chol.solve(r)
  }
}
impl iterative::SelfAdjoint for DirectInverse {}
