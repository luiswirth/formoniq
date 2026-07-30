use crate::{InnerProductSpace, LinearOperator, Report, SelfAdjoint, StopCriterion, trivial_solve};

/// Solve $A x = b$ by preconditioned conjugate gradients, started from zero.
///
/// The Krylov method for a symmetric positive-definite operator: at step $k$ it
/// returns the iterate minimizing the energy norm $norm(e)_A$ over the Krylov
/// subspace $"span"{z_0, (M^(-1)A) z_0, ...}$, reached by a three-term
/// recurrence that never stores the basis. In exact arithmetic it terminates in
/// at most $n$ steps; preconditioning by $M = B^(-1)$ compresses the spectrum so
/// far fewer are needed.
///
/// The preconditioner is taken through the [`SelfAdjoint`] bound, not
/// [`ApproxInverse`](crate::ApproxInverse): conjugate gradients is only valid
/// for a symmetric positive-definite $M$, so a one-sided sweep is rejected at
/// compile time:
///
/// ```compile_fail
/// use iterative::{krylov::cg, ApproxInverse, LinearOperator, StopCriterion, Vector};
/// // An approximate inverse that does not promise self-adjointness.
/// struct OneSided(usize);
/// impl ApproxInverse for OneSided {
///   type Space = Vector;
///   fn dim(&self) -> usize { self.0 }
///   fn apply(&self, r: &Vector) -> Vector { r.clone() }
/// }
/// fn use_it<O: LinearOperator<Space = Vector>>(a: &O, b: &Vector) {
///   // OneSided is not SelfAdjoint: this does not compile.
///   cg(a, &OneSided(b.len()), b, StopCriterion::rtol(1e-8));
/// }
/// ```
///
/// The operator's own positive-definiteness is the caller's promise, as
/// everywhere; passing an indefinite operator breaks the method (use a
/// symmetric-indefinite Krylov method for those).
pub fn cg<O: LinearOperator, M: SelfAdjoint<Space = O::Space>>(
  op: &O,
  precond: &M,
  b: &O::Space,
  stop: StopCriterion,
) -> (O::Space, Report) {
  let b_norm = b.norm();
  if b_norm == 0.0 {
    return trivial_solve(b);
  }
  let mut x = b.zeros_like();

  let mut r = b.clone();
  let mut z = precond.apply(&r);
  let mut p = z.clone();
  let mut rz = r.dot(&z);

  let mut converged;
  let mut iters = 0;
  let residual = loop {
    let residual = r.norm() / b_norm;
    converged = residual <= stop.rtol;
    // The residual check runs after every step, the nth included; the budget
    // gates only the work, so finite termination in n steps is observed.
    if converged || iters >= stop.max_iters {
      break residual;
    }
    let ap = op.apply(&p);
    let alpha = rz / p.dot(&ap);
    x.add_scaled(alpha, &p);
    r.add_scaled(-alpha, &ap);
    z = precond.apply(&r);
    let rz_next = r.dot(&z);
    let beta = rz_next / rz;
    p.scale(beta);
    p.add(&z);
    rz = rz_next;
    iters += 1;
  };
  (
    x,
    Report {
      iters,
      residual,
      converged,
    },
  )
}

/// Solve $A x = b$ by preconditioned MINRES, started from zero.
///
/// The Krylov method for a symmetric indefinite operator: it minimizes the
/// preconditioned residual norm over the Krylov subspace by a Lanczos process
/// with coupled Givens rotations, a short recurrence that never stores the
/// basis. Where [`cg`] needs $A$ positive-definite, MINRES needs only symmetry,
/// which is exactly what the mixed Hodge-Laplace saddle-point system is.
///
/// The preconditioner $M = B^(-1)$ is still taken through [`SelfAdjoint`]: MINRES
/// requires a symmetric positive-definite preconditioner (it defines the inner
/// product the residual is minimized in), even though the operator itself is
/// indefinite. In exact arithmetic it terminates in at most $n$ steps.
///
/// Follows the preconditioned form of Paige and Saunders' algorithm; the
/// reported residual is the relative preconditioner-norm residual
/// $norm(r_k)_(M^(-1)) / norm(b)_(M^(-1))$.
pub fn minres<O: LinearOperator, M: SelfAdjoint<Space = O::Space>>(
  op: &O,
  precond: &M,
  b: &O::Space,
  stop: StopCriterion,
) -> (O::Space, Report) {
  let eps = f64::EPSILON;

  // First Lanczos vector, in the M^{-1} inner product.
  let mut r1 = b.clone();
  let mut y = precond.apply(&r1);
  let beta1_sq = r1.dot(&y);
  if beta1_sq <= 0.0 {
    // b is zero. A negative value would signal a non-positive-definite
    // preconditioner, which the SelfAdjoint bound forbids.
    return trivial_solve(b);
  }
  let mut x = b.zeros_like();
  let beta1 = beta1_sq.sqrt();

  let mut oldb = 0.0;
  let mut beta = beta1;
  let mut dbar = 0.0;
  let mut epsln = 0.0;
  let mut phibar = beta1;
  let mut cs = -1.0;
  let mut sn = 0.0;
  let mut w = b.zeros_like();
  let mut w2 = b.zeros_like();
  let mut r2 = r1.clone();

  let mut residual = 1.0;
  let mut converged = false;
  let mut iters = 0;
  while iters < stop.max_iters {
    iters += 1;

    // Lanczos step in the M^{-1} inner product.
    let mut v = y.clone();
    v.scale(beta.recip());
    let mut y_next = op.apply(&v);
    if iters >= 2 {
      y_next.add_scaled(-beta / oldb, &r1);
    }
    let alfa = v.dot(&y_next);
    y_next.add_scaled(-alfa / beta, &r2);
    r1 = r2;
    r2 = y_next;
    y = precond.apply(&r2);
    oldb = beta;
    beta = r2.dot(&y).max(0.0).sqrt();

    // Apply the previous rotation, then compute and apply the next one.
    let oldeps = epsln;
    let delta = cs * dbar + sn * alfa;
    let gbar = sn * dbar - cs * alfa;
    epsln = sn * beta;
    dbar = -cs * beta;

    let gamma = (gbar * gbar + beta * beta).sqrt().max(eps);
    cs = gbar / gamma;
    sn = beta / gamma;
    let phi = cs * phibar;
    phibar *= sn;

    // Update the solution. Entering, `w` holds w_{k-1} and `w2` holds w_{k-2};
    // oldeps multiplies the older, delta the newer.
    let mut wnew = v;
    wnew.add_scaled(-oldeps, &w2);
    wnew.add_scaled(-delta, &w);
    wnew.scale(gamma.recip());
    w2 = w;
    w = wnew;
    x.add_scaled(phi, &w);

    residual = phibar / beta1;
    if residual <= stop.rtol {
      converged = true;
      break;
    }
  }
  (
    x,
    Report {
      iters,
      residual,
      converged,
    },
  )
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::testutil::{csr, dense_solve, symmetric_from_spectrum, tridiag};
  use crate::{Identity, Jacobi, Stationary, StopCriterion, Vector};

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
    use crate::testutil::symmetric_from_spectrum;
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
}

#[cfg(test)]
mod abstract_space {
  use crate::{
    ApproxInverse, InnerProductSpace, LinearOperator, SelfAdjoint, StopCriterion, Vector,
    krylov::cg, testutil::symmetric_from_spectrum,
  };

  /// A realization of the space sharing no code with nalgebra: a plain `Vec`
  /// and hand-written arithmetic.
  ///
  /// The point of the second instance is that it is a second one. If the
  /// Krylov methods still reach the same iterate here, they read nothing about
  /// their vectors beyond [`InnerProductSpace`], which is what lets the same
  /// method run on vectors that never enter host memory.
  #[derive(Clone, Debug)]
  struct Coords(Vec<f64>);

  impl InnerProductSpace for Coords {
    fn zeros_like(&self) -> Self {
      Coords(vec![0.0; self.0.len()])
    }
    fn dot(&self, other: &Self) -> f64 {
      self.0.iter().zip(&other.0).map(|(a, b)| a * b).sum()
    }
    fn scale(&mut self, alpha: f64) {
      self.0.iter_mut().for_each(|y| *y *= alpha);
    }
    fn add_scaled(&mut self, alpha: f64, x: &Self) {
      for (y, x) in self.0.iter_mut().zip(&x.0) {
        *y += alpha * x;
      }
    }
  }

  /// A dense operator over [`Coords`], row-major, applied by hand.
  struct Dense {
    rows: Vec<Vec<f64>>,
  }
  impl LinearOperator for Dense {
    type Space = Coords;
    fn dim(&self) -> usize {
      self.rows.len()
    }
    fn apply(&self, x: &Coords) -> Coords {
      Coords(
        self
          .rows
          .iter()
          .map(|row| row.iter().zip(&x.0).map(|(a, b)| a * b).sum())
          .collect(),
      )
    }
  }

  struct Unpreconditioned(usize);
  impl ApproxInverse for Unpreconditioned {
    type Space = Coords;
    fn dim(&self) -> usize {
      self.0
    }
    fn apply(&self, r: &Coords) -> Coords {
      r.clone()
    }
  }
  impl SelfAdjoint for Unpreconditioned {}

  /// The same system solved in two unrelated realizations of the space agrees,
  /// iterate for iterate.
  ///
  /// Not merely to the tolerance: conjugate gradients is a deterministic
  /// recurrence in the inner products alone, so two spaces that agree on those
  /// must produce the same iterates, and the iteration counts must match
  /// exactly. A method that peeked at an entry would have no reason to.
  #[test]
  fn the_krylov_methods_read_nothing_but_the_space() {
    let dense = symmetric_from_spectrum(&[1.0, 2.0, 3.5, 6.0, 11.0]);
    let n = dense.nrows();
    let rhs: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sqrt()).collect();

    let (host, host_report) = cg(
      &crate::testutil::csr(&dense),
      &crate::Identity::new(n),
      &Vector::from_column_slice(&rhs),
      StopCriterion::rtol(1e-12),
    );

    let op = Dense {
      rows: (0..n)
        .map(|i| dense.row(i).iter().copied().collect())
        .collect(),
    };
    let (coords, coords_report) = cg(
      &op,
      &Unpreconditioned(n),
      &Coords(rhs),
      StopCriterion::rtol(1e-12),
    );

    assert_eq!(host_report.iters, coords_report.iters);
    for (h, c) in host.iter().zip(&coords.0) {
      assert!((h - c).abs() < 1e-12, "{h} vs {c}");
    }
  }
}
