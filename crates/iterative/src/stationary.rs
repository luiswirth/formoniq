use crate::{ApproxInverse, InnerProductSpace, LinearOperator, Report, SelfAdjoint, StopCriterion};

/// Solve $A x = b$ by the stationary (preconditioned Richardson) iteration
/// $x_(k+1) = x_k + B(b - A x_k)$, started from zero.
///
/// The prototype of every method in the crate: a Krylov solve is this with
/// adaptive step coefficients, a multigrid cycle is this with $B$ the cycle
/// itself. It converges iff the spectral radius of $I - B A$ is below one, and
/// then geometrically at that rate --- global convergence, no line search, the
/// affine structure paying off. As a standalone solver it is weak (that rate is
/// mesh-dependent); its role is as the smoother and preconditioner other methods
/// wrap.
pub fn solve<S: InnerProductSpace, O: LinearOperator<S>, B: ApproxInverse<S>>(
  op: &O,
  precond: &B,
  b: &S,
  stop: StopCriterion,
) -> (S, Report) {
  let mut x = S::zeros(op.dim());
  let b_norm = b.norm().max(f64::MIN_POSITIVE);
  let mut converged;
  let mut iters = 0;
  let residual = loop {
    let mut r = b.clone();
    r.axpby(-1.0, &op.apply(&x), 1.0);
    let residual = r.norm() / b_norm;
    converged = residual <= stop.rtol;
    // Residual checked after every step and the budget gates only the work, so
    // the reported convergence reflects the final iterate, not the prior one.
    if converged || iters >= stop.max_iters {
      break residual;
    }
    x.axpby(1.0, &precond.apply(&r), 1.0);
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

/// A fixed number of stationary sweeps, packaged as an approximate inverse ---
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

impl<S: InnerProductSpace, O: LinearOperator<S>, B: ApproxInverse<S>> ApproxInverse<S>
  for Stationary<'_, O, B>
{
  fn dim(&self) -> usize {
    self.op.dim()
  }
  fn apply(&self, r: &S) -> S {
    let mut x = S::zeros(self.op.dim());
    for _ in 0..self.sweeps {
      let mut resid = r.clone();
      resid.axpby(-1.0, &self.op.apply(&x), 1.0);
      x.axpby(1.0, &self.precond.apply(&resid), 1.0);
    }
    x
  }
}

/// Self-adjoint whenever the inner preconditioner is (and the operator is
/// symmetric): each sweep is $B sum_(j<k) (I - B A)^j$, symmetric term by term
/// since $(I - B A)^j B = B (I - A B)^j$. Positive-definiteness additionally
/// needs the sweeps to converge, the constructor's promise as everywhere.
impl<S: InnerProductSpace, O: LinearOperator<S>, B: SelfAdjoint<S>> SelfAdjoint<S>
  for Stationary<'_, O, B>
{
}
