use crate::{
  InnerProductSpace, LinearOperator, RealOf, Report, ScalarOf, SelfAdjoint, StopCriterion,
  trivial_solve,
};

use approx::AbsDiffEq;
use na::{ComplexField, RealField};
use num_traits::{One, Zero};

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
///
/// Over $CC$ the hypothesis is that $A$ is *Hermitian* positive-definite,
/// $A = A^H$. A complex-*symmetric* operator $A = A^T$, which is what a lossy
/// or perfectly-matched-layer time-harmonic problem produces, is not Hermitian
/// and this method does not apply to it: it will stagnate rather than fail, so
/// the distinction is the caller's to keep.
pub fn cg<O: LinearOperator, M: SelfAdjoint<Space = O::Space>>(
  op: &O,
  precond: &M,
  b: &O::Space,
  stop: StopCriterion<RealOf<O::Space>>,
) -> (O::Space, Report<RealOf<O::Space>>) {
  let b_norm = b.norm();
  if b_norm.is_zero() {
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
  stop: StopCriterion<RealOf<O::Space>>,
) -> (O::Space, Report<RealOf<O::Space>>) {
  // Every Lanczos and rotation coefficient below is *real*, in any signature:
  // the Lanczos coefficients of a self-adjoint operator are real, and the
  // Givens rotations that follow are built from them. Only the vectors are
  // complex, and the scalars enter them through `from_real`.
  type R<O> = RealOf<<O as LinearOperator>::Space>;
  let real = |re: R<O>| ScalarOf::<O::Space>::from_real(re);
  let (zero, one) = (R::<O>::zero(), R::<O>::one());
  let eps = R::<O>::default_epsilon();

  // First Lanczos vector, in the M^{-1} inner product.
  let mut r1 = b.clone();
  let mut y = precond.apply(&r1);
  let beta1_sq = r1.dot(&y).real();
  if beta1_sq <= zero {
    // b is zero. A negative value would signal a non-positive-definite
    // preconditioner, which the SelfAdjoint bound forbids.
    return trivial_solve(b);
  }
  let mut x = b.zeros_like();
  let beta1 = beta1_sq.sqrt();

  let mut oldb = zero;
  let mut beta = beta1;
  let mut dbar = zero;
  let mut epsln = zero;
  let mut phibar = beta1;
  let mut cs = -one;
  let mut sn = zero;
  let mut w = b.zeros_like();
  let mut w2 = b.zeros_like();
  let mut r2 = r1.clone();

  let mut residual = one;
  let mut converged = false;
  let mut iters = 0;
  while iters < stop.max_iters {
    iters += 1;

    // Lanczos step in the M^{-1} inner product.
    let mut v = y.clone();
    v.scale(real(beta.recip()));
    let mut y_next = op.apply(&v);
    if iters >= 2 {
      y_next.add_scaled(real(-beta / oldb), &r1);
    }
    // Real because the operator is self-adjoint: taking the real part is that
    // hypothesis, not a discarded remainder.
    let alfa = v.dot(&y_next).real();
    y_next.add_scaled(real(-alfa / beta), &r2);
    r1 = r2;
    r2 = y_next;
    y = precond.apply(&r2);
    oldb = beta;
    beta = r2.dot(&y).real().max(zero).sqrt();

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
    wnew.add_scaled(real(-oldeps), &w2);
    wnew.add_scaled(real(-delta), &w);
    wnew.scale(real(gamma.recip()));
    w2 = w;
    w = wnew;
    x.add_scaled(real(phi), &w);

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
