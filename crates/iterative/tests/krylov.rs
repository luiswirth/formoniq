//! CG and MINRES over $RR$: CG's finite-termination theorem, preconditioning
//! preserving the fixed point, a `Stationary` sweep nested as a
//! preconditioner, and MINRES generalizing CG to the indefinite case.

mod common;

use common::{csr, dense_solve, symmetric_from_spectrum, tridiag};
use iterative::krylov::{cg, minres};
use iterative::{Identity, Jacobi, Stationary, StopCriterion, Vector};

/// CG's defining theorem: on an $n times n$ SPD system it reaches the exact
/// solution in at most $n$ steps. Swept over orders, with the degenerate
/// $n = 0, 1$ included so totality holds at the boundary. The spectrum is
/// pinned (distinct eigenvalues), since finite termination degrades under
/// ill-conditioning in floating point.
#[test]
fn cg_terminates_in_at_most_n_steps() {
  for n in 0..=8 {
    let eigs: Vec<f64> = (0..n).map(|k| 1.0 + k as f64).collect();
    let dense = symmetric_from_spectrum(&eigs);
    let a = csr(&dense);
    let x_true = Vector::from_fn(n, |i, _| (i as f64 + 1.0).ln());
    let b = &dense * &x_true;

    let stop = StopCriterion {
      rtol: 1e-10,
      max_iters: n.max(1),
    };
    let (x, report) = cg(&a, &Identity::new(n), &b, stop);
    assert!(report.converged, "n = {n} did not converge in {n} steps");
    assert!(report.iters <= n);
    if n > 0 {
      assert!((x - x_true).norm() < 1e-7, "n = {n}");
    }
  }
}

/// Preconditioning changes the path, never the fixed point: Jacobi-CG reaches
/// the same solution as unpreconditioned CG.
#[test]
fn preconditioning_preserves_the_solution() {
  let dense = tridiag(20, 4.0, 1.0);
  let a = csr(&dense);
  let x_true = Vector::from_fn(20, |i, _| ((i * i) as f64).cos());
  let b = &dense * &x_true;
  let stop = StopCriterion::rtol(1e-12);

  let (x_plain, _) = cg(&a, &Identity::new(20), &b, stop);
  let (x_jacobi, _) = cg(&a, &Jacobi::new(&a), &b, stop);
  assert!((&x_plain - &x_true).norm() < 1e-9);
  assert!((&x_jacobi - &x_true).norm() < 1e-9);
  assert!((x_plain - x_jacobi).norm() < 1e-9);
}

/// The composition that justifies the whole trait algebra: a consumer
/// (`Stationary`) used as an implementor, preconditioning another consumer
/// (`cg`). CG preconditioned by two Jacobi sweeps solves the system.
#[test]
fn cg_preconditioned_by_stationary_sweeps() {
  let dense = tridiag(20, 4.0, 1.0);
  let a = csr(&dense);
  let x_true = Vector::from_fn(20, |i, _| (i as f64 - 10.0).tanh());
  let b = &dense * &x_true;

  let sweeps = Stationary::new(&a, Jacobi::new(&a), 2);
  let (x, report) = cg(&a, &sweeps, &b, StopCriterion::rtol(1e-10));
  assert!(report.converged);
  assert!((x - x_true).norm() < 1e-7);
}

/// MINRES solves a symmetric indefinite system, the case CG cannot,
/// reproducing the direct solve, swept over orders including the degenerate
/// $n = 0, 1$.
#[test]
fn minres_solves_symmetric_indefinite_systems() {
  for n in 0..=8 {
    // A mixed-sign spectrum bounded away from zero: symmetric, indefinite,
    // nonsingular. Magnitudes 1, 1, 2, 2, ... with alternating sign.
    let eigs: Vec<f64> = (0..n)
      .map(|k| (k / 2 + 1) as f64 * if k % 2 == 0 { 1.0 } else { -1.0 })
      .collect();
    let dense = symmetric_from_spectrum(&eigs);
    let a = csr(&dense);
    let b = Vector::from_fn(n, |i, _| (i as f64 + 1.0).sqrt());

    let (x, report) = minres(&a, &Identity::new(n), &b, StopCriterion::rtol(1e-11));
    assert!(report.converged, "n = {n} did not converge");
    if n > 0 {
      assert!((x - dense_solve(&dense, &b)).norm() < 1e-7, "n = {n}");
    }
  }
}

/// On an SPD system MINRES and CG reach the same solution: MINRES is the
/// generalization, agreeing where CG applies.
#[test]
fn minres_agrees_with_cg_on_spd() {
  let dense = tridiag(25, 4.0, 1.0);
  let a = csr(&dense);
  let x_true = Vector::from_fn(25, |i, _| (i as f64).sin());
  let b = &dense * &x_true;
  let stop = StopCriterion::rtol(1e-12);

  let (x_min, report) = minres(&a, &Jacobi::new(&a), &b, stop);
  assert!(report.converged);
  assert!((&x_min - &x_true).norm() < 1e-8);

  let (x_cg, _) = cg(&a, &Jacobi::new(&a), &b, stop);
  assert!((x_min - x_cg).norm() < 1e-7);
}
