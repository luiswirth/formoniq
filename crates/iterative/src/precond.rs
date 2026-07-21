use crate::{ApproxInverse, SelfAdjoint, Vector};

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
