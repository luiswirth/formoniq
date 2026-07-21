#![doc = include_str!("../README.md")]

extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

mod operator;
mod precond;

pub use precond::{Identity, Jacobi};

/// A dense real vector, the currency of every apply.
pub type Vector = na::DVector<f64>;
/// A sparse real matrix in compressed-row storage: the assembled operator, and
/// the source a diagonal or triangular preconditioner reads its entries from.
pub type CsrMatrix = nas::CsrMatrix<f64>;

/// A linear operator $A: RR^n -> RR^n$, applied as $x |-> A x$ --- the only
/// thing a Krylov method asks of its system matrix.
///
/// Distinct from [`ApproxInverse`] by intent, not by shape: this is $A$, that is
/// $B approx A^(-1)$. Entry-needing preconditioners (a diagonal, a triangular
/// sweep) take the assembled [`CsrMatrix`] at construction instead, which keeps
/// this interface at the matrix-free minimum a matvec needs.
pub trait LinearOperator {
  /// The order $n$ of the square operator.
  fn dim(&self) -> usize;
  /// Apply the operator: $x |-> A x$.
  fn apply(&self, x: &Vector) -> Vector;
}

/// A cheap approximate inverse $B approx A^(-1)$, applied as $r |-> B r$.
///
/// One object, three roles, differing only in which consumer holds it: iterated
/// alone it is a *solver*; wrapped in a Krylov method it is a *preconditioner*;
/// sitting on a level of a multigrid hierarchy it is a *smoother*. The exact
/// inverse $B = A^(-1)$ (a factorization) is the perfect special case, and the
/// identity $B = I$ the trivial one.
pub trait ApproxInverse {
  /// The order $n$ of the square operator approximated.
  fn dim(&self) -> usize;
  /// Apply the approximate inverse: $r |-> B r$.
  fn apply(&self, r: &Vector) -> Vector;
}

/// Marker: [`ApproxInverse::apply`] is a fixed self-adjoint positive-definite
/// linear operator, $angle.l B r, s angle.r = angle.l r, B s angle.r$ and
/// $angle.l B r, r angle.r > 0$.
///
/// The precondition a symmetric Krylov method rests on. Conjugate gradients
/// takes its preconditioner only through this bound, so a non-symmetric
/// approximate inverse (a single forward Gauss-Seidel sweep, a V-cycle with
/// asymmetric smoothing) is rejected at compile time rather than converging
/// erratically at runtime. Self-adjointness is structural; positive-definiteness
/// is the constructor's promise, exactly as for the operator it approximates.
pub trait SelfAdjoint: ApproxInverse {}

/// When to stop iterating: a relative residual target and an iteration ceiling.
#[derive(Clone, Copy, Debug)]
pub struct StopCriterion {
  /// Stop once $norm(r_k) <= "rtol" dot norm(b)$.
  pub rtol: f64,
  /// Stop unconditionally after this many iterations.
  pub max_iters: usize,
}

impl StopCriterion {
  /// A relative-residual target with a generous iteration ceiling.
  pub fn rtol(rtol: f64) -> Self {
    Self {
      rtol,
      max_iters: 10_000,
    }
  }
}

/// The outcome of a solve: how far it got and whether it met the tolerance.
#[derive(Clone, Copy, Debug)]
pub struct Report {
  /// Iterations actually taken.
  pub iters: usize,
  /// The final relative residual $norm(r) / norm(b)$.
  pub residual: f64,
  /// Whether [`StopCriterion::rtol`] was met before the ceiling.
  pub converged: bool,
}

#[cfg(test)]
mod testutil {
  use crate::{CsrMatrix, Vector};
  use na::DMatrix;

  /// Sparse operator from a dense one, via triplets. Small-system test glue.
  pub fn csr(dense: &DMatrix<f64>) -> CsrMatrix {
    let (r, c) = dense.shape();
    let mut coo = nas::CooMatrix::new(r, c);
    for j in 0..c {
      for i in 0..r {
        let v = dense[(i, j)];
        if v != 0.0 {
          coo.push(i, j, v);
        }
      }
    }
    CsrMatrix::from(&coo)
  }

  /// A symmetric positive-definite operator with a prescribed spectrum,
  /// $A = Q "diag"(lambda) Q^T$ with $Q$ a deterministic orthogonal factor.
  /// Controlled conditioning: the finite-termination law degrades under an
  /// ill-conditioned random matrix, so the spectrum is pinned, not sampled.
  pub fn spd_from_spectrum(eigs: &[f64]) -> DMatrix<f64> {
    let n = eigs.len();
    let seed = DMatrix::from_fn(n, n, |i, j| ((i * 7 + j * 13) % 11) as f64 - 5.0);
    let q = seed.qr().q();
    let lambda = DMatrix::from_diagonal(&Vector::from_column_slice(eigs));
    &q * lambda * q.transpose()
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::testutil::{csr, spd_from_spectrum};
  use na::DMatrix;

  #[test]
  fn csr_matvec_matches_dense() {
    let dense = spd_from_spectrum(&[1.0, 2.0, 3.0, 4.0]);
    let a = csr(&dense);
    let x = Vector::from_column_slice(&[1.0, -2.0, 0.5, 3.0]);
    assert!((a.apply(&x) - &dense * &x).norm() < 1e-12);
    assert_eq!(LinearOperator::dim(&a), 4);
  }

  #[test]
  fn identity_applies_unchanged() {
    let id = Identity::new(3);
    let r = Vector::from_column_slice(&[2.0, -1.0, 7.0]);
    assert_eq!(id.apply(&r), r);
  }

  /// Totality at the degenerate boundary: order zero is a defined, trivial op.
  #[test]
  fn identity_is_total_at_zero() {
    let id = Identity::new(0);
    assert_eq!(id.apply(&Vector::zeros(0)), Vector::zeros(0));
  }

  /// On a diagonal operator Jacobi is the exact inverse: $B A = I$.
  #[test]
  fn jacobi_inverts_a_diagonal_operator() {
    let a = csr(&DMatrix::from_diagonal(&Vector::from_column_slice(&[
      2.0, 5.0, 0.25, 8.0,
    ])));
    let b = Jacobi::new(&a);
    let x = Vector::from_column_slice(&[1.0, -3.0, 4.0, 2.0]);
    assert!((b.apply(&a.apply(&x)) - &x).norm() < 1e-12);
  }

  /// The law the `SelfAdjoint` marker promises: $angle.l B r, s angle.r =
  /// angle.l r, B s angle.r$. Verified on a full (non-diagonal) SPD operator,
  /// whose diagonal Jacobi reads.
  #[test]
  fn jacobi_is_self_adjoint() {
    let a = csr(&spd_from_spectrum(&[1.0, 2.0, 4.0, 7.0, 9.0]));
    let b = Jacobi::new(&a);
    let r = Vector::from_column_slice(&[1.0, -2.0, 3.0, 0.5, -1.0]);
    let s = Vector::from_column_slice(&[4.0, 1.0, -1.0, 2.0, 3.0]);
    assert!((b.apply(&r).dot(&s) - r.dot(&b.apply(&s))).abs() < 1e-12);
  }
}
