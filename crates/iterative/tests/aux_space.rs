//! [`AuxiliarySpace`]: the apply is the additive sum of the smoother and
//! each pulled-back correction, the combiner is self-adjoint, and no
//! corrections is exactly the smoother.

extern crate nalgebra;

mod common;

use common::{DenseInverse, csr};
use iterative::{ApproxInverse, AuxiliarySpace, Identity, Jacobi, Vector};
use nalgebra::DMatrix;

fn spd(n: usize, seed: f64) -> DMatrix<f64> {
  let b = DMatrix::from_fn(n, n, |i, j| ((i * 7 + j * 13) as f64 * seed).sin());
  &b * b.transpose() + DMatrix::identity(n, n) * (n as f64)
}

/// The apply is exactly the additive sum: smoother plus each pulled-back solve.
#[test]
fn apply_is_the_additive_sum() {
  let n = 6;
  let m = spd(3, 0.3);
  let prolong = DMatrix::from_fn(n, 3, |i, j| ((i + 2 * j) as f64).cos());

  let b = AuxiliarySpace::new(Identity::new(n))
    .with_correction(csr(&prolong), Box::new(DenseInverse::new(&m)));

  let r = Vector::from_fn(n, |i, _| (i as f64 + 1.0).sqrt());
  let expected = &r + &prolong * m.try_inverse().unwrap() * (prolong.transpose() * &r);
  assert!((b.apply(&r) - expected).norm() < 1e-12);
}

/// $B$ is symmetric, $angle.l B r, s angle.r = angle.l r, B s angle.r$, with
/// several corrections of different shapes: the precondition CG rests on.
#[test]
fn combiner_is_self_adjoint() {
  let n = 8;
  let b = AuxiliarySpace::new(Jacobi::weighted(&csr(&spd(n, 0.5)), 0.7))
    .with_correction(
      csr(&DMatrix::from_fn(n, 4, |i, j| ((3 * i + j) as f64).sin())),
      Box::new(DenseInverse::new(&spd(4, 0.9))),
    )
    .with_correction(
      csr(&DMatrix::from_fn(n, 2, |i, j| ((i + 5 * j) as f64).cos())),
      Box::new(DenseInverse::new(&spd(2, 0.2))),
    );

  let r = Vector::from_fn(n, |i, _| (i as f64 - 3.0).tanh());
  let s = Vector::from_fn(n, |i, _| ((i * i) as f64).cos());
  assert!((b.apply(&r).dot(&s) - r.dot(&b.apply(&s))).abs() < 1e-12);
}

/// With no corrections the preconditioner is exactly its smoother: the
/// totality base case, no empty-sum special-casing.
#[test]
fn no_corrections_is_the_smoother() {
  let n = 5;
  let a = csr(&spd(n, 0.4));
  let smoother = Jacobi::weighted(&a, 0.6);
  let b = AuxiliarySpace::new(smoother.clone());
  let r = Vector::from_fn(n, |i, _| (i as f64 + 0.5).ln());
  assert!((b.apply(&r) - smoother.apply(&r)).norm() < 1e-14);
}
