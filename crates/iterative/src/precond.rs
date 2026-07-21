use crate::{ApproxInverse, CsrMatrix, SelfAdjoint, Vector};

/// The trivial approximate inverse $B = I$: apply is the identity.
///
/// Unpreconditioned iteration is preconditioned iteration with this, so it is
/// what makes an unpreconditioned Krylov solve a special case rather than a
/// separate code path. Self-adjoint by construction, and the totality base case
/// (it is defined at every order, including zero).
#[derive(Clone, Copy, Debug)]
pub struct Identity {
  dim: usize,
}

impl Identity {
  pub fn new(dim: usize) -> Self {
    Self { dim }
  }
}

impl ApproxInverse for Identity {
  fn dim(&self) -> usize {
    self.dim
  }
  fn apply(&self, r: &Vector) -> Vector {
    r.clone()
  }
}

impl SelfAdjoint for Identity {}

/// The Jacobi approximate inverse $B = D^(-1)$, the reciprocal of the diagonal.
///
/// The cheapest non-trivial approximate inverse, and the archetypal *smoother*:
/// applied to a symmetric operator it damps the high-frequency error modes and
/// leaves the low-frequency ones nearly untouched, which is exactly what a
/// multigrid level asks of it and exactly why it is a poor standalone solver. On
/// a diagonal operator it is the exact inverse. Self-adjoint whenever the
/// diagonal is positive, as it is for a positive-definite operator.
#[derive(Clone, Debug)]
pub struct Jacobi {
  inv_diag: Vector,
}

impl Jacobi {
  /// Read the diagonal of the assembled operator and invert it.
  ///
  /// Panics on a zero diagonal entry: $D^(-1)$ does not exist, and a
  /// positive-definite operator has none.
  pub fn new(a: &CsrMatrix) -> Self {
    let n = a.nrows();
    let mut diag = Vector::zeros(n);
    for (i, j, &v) in a.triplet_iter() {
      if i == j {
        diag[i] = v;
      }
    }
    assert!(
      diag.iter().all(|&d| d != 0.0),
      "Jacobi needs a nonzero diagonal"
    );
    Self {
      inv_diag: diag.map(|d| 1.0 / d),
    }
  }
}

impl ApproxInverse for Jacobi {
  fn dim(&self) -> usize {
    self.inv_diag.len()
  }
  fn apply(&self, r: &Vector) -> Vector {
    r.component_mul(&self.inv_diag)
  }
}

impl SelfAdjoint for Jacobi {}
