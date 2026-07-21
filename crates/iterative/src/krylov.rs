use crate::{LinearOperator, Report, SelfAdjoint, StopCriterion, Vector};

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
///   fn dim(&self) -> usize { self.0 }
///   fn apply(&self, r: &Vector) -> Vector { r.clone() }
/// }
/// fn use_it<O: LinearOperator>(a: &O, b: &Vector) {
///   // OneSided is not SelfAdjoint: this does not compile.
///   cg(a, &OneSided(b.len()), b, StopCriterion::rtol(1e-8));
/// }
/// ```
///
/// The operator's own positive-definiteness is the caller's promise, as
/// everywhere; passing an indefinite operator breaks the method (use a
/// symmetric-indefinite Krylov method for those).
pub fn cg<O: LinearOperator, M: SelfAdjoint>(
  op: &O,
  precond: &M,
  b: &Vector,
  stop: StopCriterion,
) -> (Vector, Report) {
  let mut x = Vector::zeros(op.dim());
  let b_norm = b.norm();
  if b_norm == 0.0 {
    return (
      x,
      Report {
        iters: 0,
        residual: 0.0,
        converged: true,
      },
    );
  }

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
    x.axpy(alpha, &p, 1.0);
    r.axpy(-alpha, &ap, 1.0);
    z = precond.apply(&r);
    let rz_next = r.dot(&z);
    let beta = rz_next / rz;
    p = &z + beta * &p;
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
