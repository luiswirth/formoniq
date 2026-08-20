use crate::{
  ApproxInverse, InnerProductSpace, LinearOperator, RealOf, Report, ScalarOf, SelfAdjoint,
  StopCriterion, trivial_solve,
};

use num_traits::{One, Zero};

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
  stop: StopCriterion<RealOf<O::Space>>,
) -> (O::Space, Report<RealOf<O::Space>>) {
  let b_norm = b.norm();
  if b_norm.is_zero() {
    return trivial_solve(b);
  }
  let mut x = b.zeros_like();
  let mut converged;
  let mut iters = 0;
  let residual = loop {
    let mut r = op.apply(&x);
    r.scale(-ScalarOf::<O::Space>::one());
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
    residual.scale(-ScalarOf::<O::Space>::one());
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
