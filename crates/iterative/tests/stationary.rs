//! Stationary iteration: Jacobi converges at the predicted geometric rate,
//! and a fixed sweep count is itself self-adjoint.

extern crate nalgebra as na;

mod common;

use common::{csr, tridiag};
use iterative::stationary::{Stationary, solve};
use iterative::{ApproxInverse, Jacobi, StopCriterion, Vector};
use na::DMatrix;

/// Stationary Jacobi iteration converges to the true solution on a
/// diagonally dominant SPD system, at a rate set by $rho(I - D^(-1) A)$.
#[test]
fn stationary_converges_to_the_solution() {
  let dense = tridiag(8, 4.0, 1.0);
  let a = csr(&dense);
  let x_true = Vector::from_fn(8, |i, _| (i as f64 - 3.5).sin());
  let b = &dense * &x_true;

  let (x, report) = solve(&a, &Jacobi::new(&a), &b, StopCriterion::rtol(1e-10));
  assert!(report.converged);
  assert!((x - x_true).norm() < 1e-8);

  // The iteration count is governed by the spectral radius, not free: the
  // predicted geometric rate bounds it (with slack for the 2-norm transient).
  let n = dense.nrows();
  let dinv = DMatrix::from_diagonal(&dense.diagonal().map(|d| 1.0 / d));
  let rho = (DMatrix::identity(n, n) - dinv * &dense)
    .complex_eigenvalues()
    .iter()
    .map(|c| c.norm())
    .fold(0.0, f64::max);
  let predicted = (1e-10_f64.ln() / rho.ln()).ceil() as usize;
  assert!(rho < 1.0 && report.iters <= 3 * predicted + 10);
}

/// A fixed number of Jacobi sweeps is itself self-adjoint, the promise the
/// `SelfAdjoint for Stationary` impl makes, and the basis of nesting it inside
/// a Krylov method.
#[test]
fn stationary_sweeps_are_self_adjoint() {
  let a = csr(&tridiag(6, 4.0, 1.0));
  let sweeps = Stationary::new(&a, Jacobi::new(&a), 3);
  let r = Vector::from_fn(6, |i, _| (i as f64).cos());
  let s = Vector::from_fn(6, |i, _| (2.0 * i as f64 + 1.0).sin());
  assert!((sweeps.apply(&r).dot(&s) - r.dot(&sweeps.apply(&s))).abs() < 1e-12);
}
