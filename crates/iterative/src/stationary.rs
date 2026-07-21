use crate::{ApproxInverse, LinearOperator, Report, SelfAdjoint, StopCriterion, Vector};

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
pub fn solve<O: LinearOperator, B: ApproxInverse>(
  op: &O,
  precond: &B,
  b: &Vector,
  stop: StopCriterion,
) -> (Vector, Report) {
  let mut x = Vector::zeros(op.dim());
  let b_norm = b.norm().max(f64::MIN_POSITIVE);
  let mut residual = 1.0;
  let mut converged = b.norm() == 0.0;
  let mut iters = 0;
  while iters < stop.max_iters && !converged {
    let r = b - op.apply(&x);
    residual = r.norm() / b_norm;
    if residual <= stop.rtol {
      converged = true;
      break;
    }
    x += precond.apply(&r);
    iters += 1;
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

impl<O: LinearOperator, B: ApproxInverse> ApproxInverse for Stationary<'_, O, B> {
  fn dim(&self) -> usize {
    self.op.dim()
  }
  fn apply(&self, r: &Vector) -> Vector {
    let mut x = Vector::zeros(self.op.dim());
    for _ in 0..self.sweeps {
      let resid = r - self.op.apply(&x);
      x += self.precond.apply(&resid);
    }
    x
  }
}

/// Self-adjoint whenever the inner preconditioner is (and the operator is
/// symmetric): each sweep is $B sum_(j<k) (I - B A)^j$, symmetric term by term
/// since $(I - B A)^j B = B (I - A B)^j$. Positive-definiteness additionally
/// needs the sweeps to converge, the constructor's promise as everywhere.
impl<O: LinearOperator, B: SelfAdjoint> SelfAdjoint for Stationary<'_, O, B> {}
