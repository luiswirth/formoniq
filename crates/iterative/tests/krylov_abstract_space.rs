//! The Krylov methods read nothing about their vectors beyond
//! [`InnerProductSpace`]: a second, unrelated realization of the space
//! reaches the same iterates as the nalgebra-backed one, exactly.

mod common;

use common::{csr, symmetric_from_spectrum};
use iterative::krylov::cg;
use iterative::{
  ApproxInverse, Identity, InnerProductSpace, LinearOperator, SelfAdjoint, StopCriterion, Vector,
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
    &csr(&dense),
    &Identity::new(n),
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
