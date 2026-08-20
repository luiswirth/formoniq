//! Preconditioners: [`Identity`] is a no-op, [`Jacobi`] inverts a diagonal
//! operator exactly and is self-adjoint, and [`BlockDiagonal`] inherits both
//! from its blocks.

extern crate nalgebra as na;

mod common;

use common::{csr, dense_solve, symmetric_from_spectrum};
use iterative::krylov::cg;
use iterative::{
  ApproxInverse, BlockDiagonal, Identity, Jacobi, LinearOperator, StopCriterion, Vector,
};
use na::DMatrix;

#[test]
fn identity_applies_unchanged() {
  let id = Identity::new(3);
  let r = Vector::from_column_slice(&[2.0, -1.0, 7.0]);
  assert_eq!(id.apply(&r), r);
}

/// Totality at the degenerate boundary: order zero is a defined, trivial op.
#[test]
fn identity_is_total_at_zero() {
  let id = Identity::<Vector>::new(0);
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
  let a = csr(&symmetric_from_spectrum(&[1.0, 2.0, 4.0, 7.0, 9.0]));
  let b = Jacobi::new(&a);
  let r = Vector::from_column_slice(&[1.0, -2.0, 3.0, 0.5, -1.0]);
  let s = Vector::from_column_slice(&[4.0, 1.0, -1.0, 2.0, 3.0]);
  assert!((b.apply(&r).dot(&s) - r.dot(&b.apply(&s))).abs() < 1e-12);
}

/// A block-diagonal preconditioner applies each block's inverse to its own
/// slice: for a block-diagonal operator with exact (Jacobi-on-diagonal)
/// blocks it is the exact inverse, so preconditioned CG converges in one step.
#[test]
fn block_diagonal_of_exact_blocks_is_exact() {
  // A block-diagonal operator, each block a distinct diagonal matrix.
  let sizes = [3usize, 4, 2];
  let n: usize = sizes.iter().sum();
  let diag = DMatrix::from_diagonal(&Vector::from_fn(n, |i, _| 1.0 + (i % 6) as f64));
  let a = csr(&diag);

  let mut blocks = Vec::new();
  let mut off = 0;
  for &d in &sizes {
    let sub = csr(&diag.view((off, off), (d, d)).into_owned());
    blocks.push(Jacobi::new(&sub));
    off += d;
  }
  let precond = BlockDiagonal::new(blocks);
  assert_eq!(precond.dim(), n);

  let b = Vector::from_fn(n, |i, _| (i as f64 - 4.0).cos());
  let (x, report) = cg(&a, &precond, &b, StopCriterion::rtol(1e-12));
  assert!(
    report.converged && report.iters <= 1,
    "iters = {}",
    report.iters
  );
  assert!((x - dense_solve(&diag, &b)).norm() < 1e-9);
}

/// Block-diagonal is self-adjoint when its blocks are, so it may precondition
/// CG/MINRES exactly when they may.
#[test]
fn block_diagonal_is_self_adjoint_from_blocks() {
  let a1 = csr(&symmetric_from_spectrum(&[1.0, 2.0, 4.0]));
  let a2 = csr(&symmetric_from_spectrum(&[3.0, 5.0]));
  let precond = BlockDiagonal::new(vec![Jacobi::new(&a1), Jacobi::new(&a2)]);
  let r = Vector::from_fn(5, |i, _| (i as f64 + 1.0).ln());
  let s = Vector::from_fn(5, |i, _| (2.0 * i as f64).cos());
  assert!((precond.apply(&r).dot(&s) - r.dot(&precond.apply(&s))).abs() < 1e-12);
}
