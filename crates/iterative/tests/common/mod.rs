//! Fixtures shared by the law tests of every solver in the crate: a dense
//! direct inverse to stand in for a bottom solve, deterministic SPD/tridiagonal
//! probe matrices, and the sparse conversion glue.
//!
//! Each test binary compiles this module fresh and uses only some of it,
//! which is not dead code, just an unused export from any one binary's view.
#![allow(dead_code)]

extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use iterative::{ApproxInverse, CsrMatrix, Field, SelfAdjoint, Vector};
use na::DMatrix;

/// A dense direct inverse: the exact $A^(-1)$, standing in for the
/// factorization a consumer supplies at the bottom of a V-cycle or on an
/// auxiliary space. Self-adjoint for a self-adjoint operator.
pub struct DenseInverse<T = f64> {
  inv: DMatrix<T>,
}
impl<T: Field> DenseInverse<T> {
  pub fn new(a: &DMatrix<T>) -> Self {
    Self {
      inv: a.clone().try_inverse().expect("nonsingular"),
    }
  }
}
impl<T: Field> ApproxInverse for DenseInverse<T> {
  type Space = Vector<T>;
  fn dim(&self) -> usize {
    self.inv.nrows()
  }
  fn apply(&self, r: &Vector<T>) -> Vector<T> {
    &self.inv * r
  }
}
impl<T: Field> SelfAdjoint for DenseInverse<T> {}

/// Sparse operator from a dense one, via triplets. Small-system test glue.
pub fn csr<T: Field>(dense: &DMatrix<T>) -> CsrMatrix<T> {
  let (r, c) = dense.shape();
  let mut coo = nas::CooMatrix::new(r, c);
  for j in 0..c {
    for i in 0..r {
      let v = dense[(i, j)];
      if !v.is_zero() {
        coo.push(i, j, v);
      }
    }
  }
  CsrMatrix::from(&coo)
}

/// A symmetric operator with a prescribed spectrum,
/// $A = Q "diag"(lambda) Q^T$ with $Q$ a deterministic orthogonal factor.
/// Positive definiteness is the caller's choice of $lambda$, which is also
/// what separates the CG-shaped case from the MINRES-shaped one.
///
/// Controlled conditioning: the finite-termination law degrades under an
/// ill-conditioned random matrix, so the spectrum is pinned, not sampled.
pub fn symmetric_from_spectrum(eigs: &[f64]) -> DMatrix<f64> {
  let n = eigs.len();
  let seed = DMatrix::from_fn(n, n, |i, j| ((i * 7 + j * 13) % 11) as f64 - 5.0);
  let q = seed.qr().q();
  let lambda = DMatrix::from_diagonal(&Vector::from_column_slice(eigs));
  &q * lambda * q.transpose()
}

/// The tridiagonal $"diag" I - "off" (L + L^T)$: SPD, and strictly diagonally
/// dominant for $"diag" > 2 "off"$, so a Jacobi sweep is a contraction.
pub fn tridiag(n: usize, diag: f64, off: f64) -> DMatrix<f64> {
  DMatrix::from_fn(n, n, |i, j| {
    if i == j {
      diag
    } else if i.abs_diff(j) == 1 {
      -off
    } else {
      0.0
    }
  })
}

/// Direct dense solve, the reference an iterative method must reproduce.
pub fn dense_solve<T: Field>(a: &DMatrix<T>, b: &Vector<T>) -> Vector<T> {
  a.clone().lu().solve(b).expect("nonsingular")
}

/// An orthonormal basis of the Krylov subspace
/// $K_k (A, b) = "span"{b, A b, ..., A^(k-1) b}$, the space every Krylov
/// iterate lies in and the minimization laws quantify over.
///
/// Orthonormalized (twice-applied Gram-Schmidt) rather than held as the raw
/// powers: the power basis is exponentially ill-conditioned, and the law
/// compares a minimum computed in this basis against the solver's iterate.
/// Fewer than `k` columns come back once the space saturates, which is the
/// honest answer there and not a truncation.
pub fn krylov_basis(a: &DMatrix<f64>, b: &Vector, k: usize) -> DMatrix<f64> {
  let mut cols: Vec<Vector> = Vec::new();
  let mut next = b.clone();
  for _ in 0..k {
    let mut w = next.clone();
    for _ in 0..2 {
      for q in &cols {
        let c = q.dot(&w);
        w -= q * c;
      }
    }
    let norm = w.norm();
    if norm < 1e-10 {
      break;
    }
    cols.push(w / norm);
    next = a * cols.last().unwrap();
  }
  if cols.is_empty() {
    DMatrix::zeros(b.len(), 0)
  } else {
    DMatrix::from_columns(&cols)
  }
}
