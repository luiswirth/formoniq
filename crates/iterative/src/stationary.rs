use crate::{
  ApproxInverse, InnerProductSpace, LinearOperator, Report, SelfAdjoint, StopCriterion,
  trivial_solve,
};

/// Solve $A x = b$ by the stationary (preconditioned Richardson) iteration
/// $x_(k+1) = x_k + B(b - A x_k)$, started from zero.
///
/// The prototype of every method in the crate: a Krylov solve is this with
/// adaptive step coefficients, a multigrid cycle is this with $B$ the cycle
/// itself. It converges iff the spectral radius of $I - B A$ is below one, and
/// then geometrically at that rate, global convergence, no line search, the
/// affine structure paying off. As a standalone solver it is weak (that rate is
/// mesh-dependent). Its role is as the smoother and preconditioner other methods
/// wrap.
pub fn solve<O: LinearOperator, B: ApproxInverse<Space = O::Space>>(
  op: &O,
  precond: &B,
  b: &O::Space,
  stop: StopCriterion,
) -> (O::Space, Report) {
  let b_norm = b.norm();
  if b_norm == 0.0 {
    return trivial_solve(b);
  }
  let mut x = b.zeros_like();
  let mut converged;
  let mut iters = 0;
  let residual = loop {
    let mut r = op.apply(&x);
    r.scale(-1.0);
    r.add(b);
    let residual = r.norm() / b_norm;
    converged = residual <= stop.rtol;
    // Residual checked after every step and the budget gates only the work, so
    // the reported convergence reflects the final iterate, not the prior one.
    if converged || iters >= stop.max_iters {
      break residual;
    }
    x.add(&precond.apply(&r));
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

/// Refine `x` toward solving $A x = b$ by `count` stationary steps
/// $x <- x + B(b - A x)$, continuing from the incoming `x`.
///
/// The one place the stationary step is written. A [`Stationary`] preconditioner
/// is this started from zero, and a multigrid level's smoothing is this
/// continuing from the iterate the coarse correction left behind.
pub fn sweeps<O: LinearOperator, B: ApproxInverse<Space = O::Space>>(
  op: &O,
  precond: &B,
  b: &O::Space,
  x: &mut O::Space,
  count: usize,
) {
  for _ in 0..count {
    let mut residual = op.apply(x);
    residual.scale(-1.0);
    residual.add(b);
    x.add(&precond.apply(&residual));
  }
}

/// A fixed number of stationary sweeps, packaged as an approximate inverse,
/// the same object as [`solve`], read as a preconditioner rather than a solver.
///
/// This is what makes the crate compose: a consumer is itself an implementor, so
/// `k` Jacobi sweeps become a preconditioner for a Krylov method, exactly the
/// pattern a multigrid V-cycle will follow. Borrows the operator, since a
/// preconditioner is tied to the system it approximates.
#[derive(Clone, Copy, Debug)]
pub struct Stationary<'a, O, B> {
  op: &'a O,
  precond: B,
  sweeps: usize,
}

impl<'a, O: LinearOperator, B: ApproxInverse> Stationary<'a, O, B> {
  /// `sweeps` applications of `precond` toward inverting `op`.
  pub fn new(op: &'a O, precond: B, sweeps: usize) -> Self {
    Self {
      op,
      precond,
      sweeps,
    }
  }
}

impl<O: LinearOperator, B: ApproxInverse<Space = O::Space>> ApproxInverse for Stationary<'_, O, B> {
  type Space = O::Space;
  fn dim(&self) -> usize {
    self.op.dim()
  }
  fn apply(&self, r: &Self::Space) -> Self::Space {
    let mut x = r.zeros_like();
    sweeps(self.op, &self.precond, r, &mut x, self.sweeps);
    x
  }
}

/// Self-adjoint whenever the inner preconditioner is (and the operator is
/// symmetric): each sweep is $B sum_(j<k) (I - B A)^j$, symmetric term by term
/// since $(I - B A)^j B = B (I - A B)^j$. Positive-definiteness additionally
/// needs the sweeps to converge, the constructor's promise as everywhere.
impl<O: LinearOperator, B: SelfAdjoint<Space = O::Space>> SelfAdjoint for Stationary<'_, O, B> {}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::testutil::{csr, tridiag};
  use crate::{Jacobi, Vector};
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
}
