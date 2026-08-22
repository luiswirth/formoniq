//! [`BlockDiagonal`]: on a block-diagonal operator whose blocks are inverted
//! exactly it is the exact inverse, which is the identity characterizing it.

extern crate nalgebra as na;

mod common;

use common::{csr, dense_solve};
use iterative::krylov::cg;
use iterative::{BlockDiagonal, Jacobi, StopCriterion, Vector};
use na::DMatrix;

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

  let b = Vector::from_fn(n, |i, _| (i as f64 - 4.0).cos());
  let (x, report) = cg(&a, &precond, &b, StopCriterion::rtol(1e-12));
  assert!(
    report.converged && report.iters <= 1,
    "iters = {}",
    report.iters
  );
  assert!((x - dense_solve(&diag, &b)).norm() < 1e-9);
}
