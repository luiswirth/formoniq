//! Stationary iteration: the error contracts by the iteration matrix at every
//! sweep, so convergence is geometric at the rate its spectral radius sets.

extern crate nalgebra as na;

mod common;

use common::{csr, tridiag};
use iterative::stationary::solve;
use iterative::{Jacobi, StopCriterion, Vector};
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
