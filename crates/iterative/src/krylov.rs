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

/// The same laws over $CC$, where a misplaced conjugate is visible.
///
/// Conjugation is the identity on $RR$, so every test above passes on an
/// implementation whose inner product is bilinear rather than Hermitian, or
/// whose adjoint is a bare transpose. None of them can tell the difference,
/// which is exactly why these exist: the complex case is not an extra feature
/// being checked, it is the only place the convention is observable.
#[cfg(test)]
mod complex {
  use super::*;
  use crate::testutil::{csr, dense_solve};
  use crate::{Identity, InnerProductSpace, Jacobi, StopCriterion, Vector, adjoint};
  use na::{Complex, DMatrix};

  type C = Complex<f64>;

  fn c(re: f64, im: f64) -> C {
    Complex::new(re, im)
  }

  /// A Hermitian matrix with a prescribed (necessarily real) spectrum,
  /// $A = Q Lambda Q^H$ with $Q$ a deterministic unitary factor.
  ///
  /// Genuinely complex: $Q$ has a nonzero imaginary part, so $A^T != A^H$ and
  /// the two conventions disagree on it.
  fn hermitian_from_spectrum(eigs: &[f64]) -> DMatrix<C> {
    let n = eigs.len();
    let seed = DMatrix::from_fn(n, n, |i, j| {
      c(
        ((i * 7 + j * 13) % 11) as f64 - 5.0,
        ((i * 5 + j * 3) % 7) as f64 - 3.0,
      )
    });
    let q = seed.qr().q();
    let lambda = DMatrix::from_diagonal(&Vector::from_iterator(n, eigs.iter().map(|&e| c(e, 0.0))));
    &q * lambda * q.adjoint()
  }

  fn rhs(n: usize) -> Vector<C> {
    Vector::from_fn(n, |i, _| c((i as f64 + 1.0).sqrt(), (i as f64 - 2.0).cos()))
  }

  /// The inner product is sesquilinear, conjugate-linear in its first argument:
  /// $angle.l i x, y angle.r = -i angle.l x, y angle.r$ and
  /// $angle.l x, i y angle.r = i angle.l x, y angle.r$.
  ///
  /// The two halves must be checked separately. A bilinear `dot` satisfies
  /// neither, and a `dot` conjugating the *other* argument satisfies both with
  /// the signs exchanged, which is the mistake a single-sided test misses.
  #[test]
  fn the_inner_product_is_conjugate_linear_in_its_first_argument() {
    // Spelled through the trait, never as `x.dot(&y)`: nalgebra's inherent
    // `dot` is the *bilinear* product and wins method resolution on a concrete
    // vector, so the shorthand would test nalgebra rather than the trait. The
    // generic code cannot make this mistake, having no inherent method to find.
    let dot = InnerProductSpace::dot;
    let (x, y) = (
      rhs(5),
      Vector::from_fn(5, |i, _| c((i as f64).sin(), 1.0 - i as f64)),
    );
    let xy = dot(&x, &y);
    let i = c(0.0, 1.0);

    assert!((dot(&(x.clone() * i), &y) - (-i) * xy).norm() < 1e-12);
    assert!((dot(&x, &(y.clone() * i)) - i * xy).norm() < 1e-12);
    // And it is positive definite, so the induced norm is real.
    assert!(dot(&x, &x).im.abs() < 1e-12 && dot(&x, &x).re > 0.0);
  }

  /// The adjoint is the conjugate transpose, $(A^H)_(i j) = overline(A_(j i))$,
  /// and it is what makes $angle.l A x, y angle.r = angle.l x, A^H y angle.r$.
  /// The bare transpose satisfies neither over $CC$.
  #[test]
  fn the_adjoint_is_the_conjugate_transpose() {
    let dense = DMatrix::from_fn(4, 3, |i, j| c(i as f64 - 1.0, 2.0 * j as f64 - 1.0));
    let a = csr(&dense);
    let dot = InnerProductSpace::dot;
    let (x, y) = (rhs(3), rhs(4));
    let ax = &dense * &x;
    let ahy = &DMatrix::from(&adjoint(&a)) * &y;
    assert!((dot(&ax, &y) - dot(&x, &ahy)).norm() < 1e-12);
  }

  /// CG's defining theorem over $CC$: on an $n times n$ Hermitian
  /// positive-definite system it reaches the exact solution in at most $n$
  /// steps. Swept over orders with the degenerate $n = 0, 1$ included.
  ///
  /// This is the test a bilinear inner product fails: the recurrence is no
  /// longer conjugate-orthogonal, so it neither terminates nor converges.
  #[test]
  fn cg_terminates_on_a_hermitian_positive_definite_system() {
    for n in 0..=8 {
      let eigs: Vec<f64> = (0..n).map(|k| 1.0 + k as f64).collect();
      let dense = hermitian_from_spectrum(&eigs);
      let a = csr(&dense);
      let b = rhs(n);

      let stop = StopCriterion {
        rtol: 1e-10,
        max_iters: n.max(1),
      };
      let (x, report) = cg(&a, &Identity::new(n), &b, stop);
      assert!(report.converged, "n = {n} did not converge in {n} steps");
      if n > 0 {
        assert!((x - dense_solve(&dense, &b)).norm() < 1e-7, "n = {n}");
      }
    }
  }

  /// MINRES solves a Hermitian *indefinite* complex system, the case CG cannot,
  /// reproducing the direct solve. Its Lanczos and rotation coefficients are
  /// real throughout, which is what self-adjointness buys.
  #[test]
  fn minres_solves_a_hermitian_indefinite_system() {
    for n in 0..=8 {
      let eigs: Vec<f64> = (0..n)
        .map(|k| (k / 2 + 1) as f64 * if k % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
      let dense = hermitian_from_spectrum(&eigs);
      let a = csr(&dense);
      let b = rhs(n);

      let (x, report) = minres(&a, &Identity::new(n), &b, StopCriterion::rtol(1e-11));
      assert!(report.converged, "n = {n} did not converge");
      if n > 0 {
        assert!((x - dense_solve(&dense, &b)).norm() < 1e-7, "n = {n}");
      }
    }
  }

  /// Preconditioning a complex system changes the path, never the fixed point.
  /// Jacobi reads a Hermitian operator's diagonal, which is real.
  #[test]
  fn preconditioning_preserves_the_complex_solution() {
    let dense = hermitian_from_spectrum(&[1.0, 2.0, 3.5, 6.0, 11.0, 14.0]);
    let a = csr(&dense);
    let b = rhs(6);
    let stop = StopCriterion::rtol(1e-12);

    let (x_plain, _) = cg(&a, &Identity::new(6), &b, stop);
    let (x_jacobi, _) = cg(&a, &Jacobi::new(&a), &b, stop);
    assert!((&x_plain - dense_solve(&dense, &b)).norm() < 1e-9);
    assert!((x_plain - x_jacobi).norm() < 1e-9);
  }

  /// A real system embedded in $CC$ has the real solution: extension of scalars
  /// commutes with the solve, so the complex instantiation is a generalization
  /// of the real one rather than a parallel implementation of it.
  #[test]
  fn a_real_system_solved_over_the_complexes_stays_real() {
    let dense = crate::testutil::symmetric_from_spectrum(&[1.0, 2.0, 4.0, 7.0, 9.0]);
    let b = Vector::from_fn(5, |i, _| (i as f64 + 1.0).ln());
    let stop = StopCriterion::rtol(1e-12);
    let (x_real, _) = cg(&csr(&dense), &Identity::new(5), &b, stop);

    let dense_c = dense.map(|v| c(v, 0.0));
    let (x_complex, _) = cg(
      &csr(&dense_c),
      &Identity::new(5),
      &b.map(|v| c(v, 0.0)),
      StopCriterion::rtol(1e-12),
    );
    assert!(x_complex.iter().all(|z| z.im.abs() < 1e-12));
    assert!((x_complex.map(|z| z.re) - x_real).norm() < 1e-12);
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
    type Scalar = f64;
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
