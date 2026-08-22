//! CG and MINRES over $RR$, by their defining properties: at every step the
//! iterate is the minimizer over the Krylov subspace, of the energy norm of
//! the error for CG and of the residual norm for MINRES. With CG's
//! finite-termination theorem, preconditioning preserving the fixed point,
//! and MINRES solving the symmetric indefinite case CG cannot.

extern crate nalgebra as na;

mod common;

use common::{csr, dense_solve, krylov_basis, symmetric_from_spectrum, tridiag};
use iterative::krylov::{cg, minres};
use iterative::{CsrMatrix, Identity, Jacobi, SelfAdjoint, Stationary, StopCriterion, Vector};
use na::DMatrix;

/// $norm(e)_A = sqrt(chevron.l e, A e chevron.r)$, the energy norm CG minimizes.
fn energy_norm(a: &DMatrix<f64>, e: &Vector) -> f64 {
  e.dot(&(a * e)).sqrt()
}

/// Run a method for exactly `k` steps: an unreachable tolerance turns the
/// iteration budget into a step count, so the $k$-th iterate is what comes
/// back.
fn exactly(k: usize) -> StopCriterion {
  StopCriterion {
    rtol: 0.0,
    max_iters: k,
  }
}

/// CG's defining property: the $k$-th iterate minimizes the energy norm of the
/// error over the Krylov subspace,
/// $ x_k = op("argmin")_(x in K_k (A, b)) norm(x^* - x)_A, $
/// with $K_k (A, b) = "span"{b, A b, ..., A^(k-1) b}$.
///
/// The one property that distinguishes CG from any other descent method:
/// steepest descent, a Chebyshev iteration and a mistuned $beta$ all produce
/// iterates in that same subspace, and none of them attains the minimum over
/// it. The minimum is computed independently, by solving the projected system
/// $V^T A V y = V^T b$ in an orthonormal basis $V$ of the subspace.
#[test]
fn cg_minimizes_the_energy_norm_over_the_krylov_subspace() {
  for n in 1..=6 {
    let eigs: Vec<f64> = (0..n).map(|k| 1.0 + k as f64).collect();
    let dense = symmetric_from_spectrum(&eigs);
    let a = csr(&dense);
    let x_true = Vector::from_fn(n, |i, _| (i as f64 + 1.0).ln() + 0.5);
    let b = &dense * &x_true;

    for k in 1..=n {
      let (x, report) = cg(&a, &Identity::new(n), &b, exactly(k));
      assert_eq!(report.iters, k, "n = {n}, k = {k}");

      let v = krylov_basis(&dense, &b, k);
      let y = (v.transpose() * &dense * &v)
        .lu()
        .solve(&(v.transpose() * &b))
        .expect("projected system");
      let minimal = energy_norm(&dense, &(&x_true - &v * y));

      let attained = energy_norm(&dense, &(&x_true - &x));
      assert!(
        (attained - minimal).abs() < 1e-9 * (1.0 + minimal),
        "n = {n}, k = {k}: CG attained {attained}, the minimum over the subspace is {minimal}"
      );
    }
  }
}

/// MINRES's defining property, the symmetric-indefinite analogue: the $k$-th
/// iterate minimizes the residual norm over the Krylov subspace,
/// $ x_k = op("argmin")_(x in K_k (A, b)) norm(b - A x), $
/// which asks only that $A$ be self-adjoint, never that it be definite.
///
/// The minimum is computed independently, as the least-squares problem
/// $min_y norm(b - A V y)$ in an orthonormal basis $V$ of the subspace.
#[test]
fn minres_minimizes_the_residual_over_the_krylov_subspace() {
  for n in 1..=6 {
    let eigs: Vec<f64> = (0..n)
      .map(|k| (k / 2 + 1) as f64 * if k % 2 == 0 { 1.0 } else { -1.0 })
      .collect();
    let dense = symmetric_from_spectrum(&eigs);
    let a = csr(&dense);
    let b = Vector::from_fn(n, |i, _| (i as f64 + 1.0).sqrt());

    for k in 1..=n {
      let (x, _) = minres(&a, &Identity::new(n), &b, exactly(k));

      let w = &dense * krylov_basis(&dense, &b, k);
      let y = (w.transpose() * &w)
        .lu()
        .solve(&(w.transpose() * &b))
        .expect("normal equations");
      let minimal = (&b - &w * y).norm();

      let attained = (&b - &dense * &x).norm();
      assert!(
        (attained - minimal).abs() < 1e-9 * (1.0 + minimal),
        "n = {n}, k = {k}: MINRES attained {attained}, the minimum over the subspace is {minimal}"
      );
    }
  }
}

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

/// Preconditioning changes the path, never the fixed point: $B A x = B b$ has
/// the solution of $A x = b$ for any invertible $B$, so every self-adjoint
/// positive-definite choice reaches the same $x$.
///
/// Swept over the crate's preconditioners, the composite one included: a
/// fixed number of stationary sweeps is itself an approximate inverse, which
/// is the nesting the trait algebra exists for.
#[test]
fn preconditioning_preserves_the_solution() {
  let n = 20;
  let dense = tridiag(n, 4.0, 1.0);
  let a = csr(&dense);
  let x_true = Vector::from_fn(n, |i, _| ((i * i) as f64).cos());
  let b = &dense * &x_true;
  let stop = StopCriterion::rtol(1e-12);

  let reaches_the_solution = |precond: &dyn SelfAdjointProbe| precond.check(&a, &b, &x_true, stop);
  reaches_the_solution(&Identity::new(n));
  reaches_the_solution(&Jacobi::new(&a));
  reaches_the_solution(&Jacobi::weighted(&a, 0.7));
  reaches_the_solution(&Stationary::new(&a, Jacobi::new(&a), 2));
}

/// Erased driver for the sweep above: both methods run against one
/// preconditioner, the concrete type staying with the implementor.
trait SelfAdjointProbe {
  fn check(&self, a: &CsrMatrix, b: &Vector, x_true: &Vector, stop: StopCriterion);
}
impl<B: SelfAdjoint<Space = Vector>> SelfAdjointProbe for B {
  fn check(&self, a: &CsrMatrix, b: &Vector, x_true: &Vector, stop: StopCriterion) {
    let (x, report) = cg(a, self, b, stop);
    assert!(report.converged);
    assert!((&x - x_true).norm() < 1e-9);

    let (x, report) = minres(a, self, b, stop);
    assert!(report.converged);
    assert!((&x - x_true).norm() < 1e-9);
  }
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
