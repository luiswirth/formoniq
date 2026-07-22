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

/// Solve $A x = b$ by preconditioned MINRES, started from zero.
///
/// The Krylov method for a symmetric *indefinite* operator: it minimizes the
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
pub fn minres<O: LinearOperator, M: SelfAdjoint>(
  op: &O,
  precond: &M,
  b: &Vector,
  stop: StopCriterion,
) -> (Vector, Report) {
  let n = op.dim();
  let mut x = Vector::zeros(n);
  let eps = f64::EPSILON;

  // First Lanczos vector, in the M^{-1} inner product.
  let mut r1 = b.clone();
  let mut y = precond.apply(&r1);
  let beta1_sq = r1.dot(&y);
  if beta1_sq <= 0.0 {
    // b is zero (nothing to solve); a negative value would signal a
    // non-positive-definite preconditioner, which the SelfAdjoint bound forbids.
    return (
      x,
      Report {
        iters: 0,
        residual: 0.0,
        converged: true,
      },
    );
  }
  let beta1 = beta1_sq.sqrt();

  let mut oldb = 0.0;
  let mut beta = beta1;
  let mut dbar = 0.0;
  let mut epsln = 0.0;
  let mut phibar = beta1;
  let mut cs = -1.0;
  let mut sn = 0.0;
  let mut w = Vector::zeros(n);
  let mut w2 = Vector::zeros(n);
  let mut r2 = r1.clone();

  let mut residual = 1.0;
  let mut converged = false;
  let mut iters = 0;
  while iters < stop.max_iters {
    iters += 1;

    // Lanczos step in the M^{-1} inner product.
    let v = &y / beta;
    let mut y_next = op.apply(&v);
    if iters >= 2 {
      y_next.axpy(-beta / oldb, &r1, 1.0);
    }
    let alfa = v.dot(&y_next);
    y_next.axpy(-alfa / beta, &r2, 1.0);
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
    wnew.axpy(-oldeps, &w2, 1.0);
    wnew.axpy(-delta, &w, 1.0);
    wnew /= gamma;
    w2 = w;
    w = wnew;
    x.axpy(phi, &w, 1.0);

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
