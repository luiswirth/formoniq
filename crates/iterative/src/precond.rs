use crate::{ApproxInverse, CsrMatrix, Field, InnerProductSpace, SelfAdjoint, Vector};

use num_traits::One;
use std::marker::PhantomData;

/// The trivial approximate inverse $B = I$: apply is the identity.
///
/// Unpreconditioned iteration is preconditioned iteration with this, so it is
/// what makes an unpreconditioned Krylov solve a special case rather than a
/// separate code path. Self-adjoint by construction, and the totality base case
/// (it is defined at every order, including zero).
///
/// Generic over the space rather than fixed to [`Vector`]: the identity is
/// defined wherever an [`InnerProductSpace`] is, and pinning it to one would
/// make an unpreconditioned solve on any other space impossible.
#[derive(Clone, Copy, Debug)]
pub struct Identity<S = Vector> {
  dim: usize,
  space: PhantomData<S>,
}

impl<S> Identity<S> {
  pub fn new(dim: usize) -> Self {
    Self {
      dim,
      space: PhantomData,
    }
  }
}

impl<S: InnerProductSpace> ApproxInverse for Identity<S> {
  type Space = S;
  fn dim(&self) -> usize {
    self.dim
  }
  fn apply(&self, r: &S) -> S {
    r.clone()
  }
}

impl<S: InnerProductSpace> SelfAdjoint for Identity<S> {}

/// The Jacobi approximate inverse $B = D^(-1)$, the reciprocal of the diagonal.
///
/// The cheapest non-trivial approximate inverse, and the archetypal smoother:
/// applied to a symmetric operator it damps the high-frequency error modes and
/// leaves the low-frequency ones nearly untouched, which is exactly what a
/// multigrid level asks of it and exactly why it is a poor standalone solver. On
/// a diagonal operator it is the exact inverse. Self-adjoint whenever the
/// diagonal is positive, as it is for a positive-definite operator: over $CC$
/// that means real and positive, which the diagonal of a Hermitian operator is.
#[derive(Clone, Debug)]
pub struct Jacobi<T = f64> {
  inv_diag: Vector<T>,
}

impl<T: Field> Jacobi<T> {
  /// Read the diagonal of the assembled operator and invert it.
  ///
  /// Panics on a zero diagonal entry: $D^(-1)$ does not exist, and a
  /// positive-definite operator has none.
  pub fn new(a: &CsrMatrix<T>) -> Self {
    Self::weighted(a, T::RealField::one())
  }

  /// The weighted (damped) Jacobi inverse $B = omega D^(-1)$.
  ///
  /// Unit weight is the plain Jacobi inverse, exact on a diagonal operator. A
  /// sub-unit weight is what makes Jacobi a smoother: undamped Jacobi barely
  /// touches the highest-frequency error of a second-order operator (its
  /// iteration matrix has eigenvalue near $-1$ there), while $omega approx 2\/3$
  /// damps the whole upper half of the spectrum, which is exactly the error a
  /// multigrid level must remove before coarsening. Self-adjoint for any
  /// $omega > 0$ on a positive diagonal.
  ///
  /// The weight is real: it is a damping factor, and a complex one would turn
  /// the smoother into something that is not self-adjoint.
  pub fn weighted(a: &CsrMatrix<T>, omega: T::RealField) -> Self {
    let n = a.nrows();
    let mut diag = Vector::zeros(n);
    for (i, j, v) in a.triplet_iter() {
      if i == j {
        diag[i] = *v;
      }
    }
    Self::from_diagonal(&diag, omega)
  }

  /// The Jacobi *smoother*: the weight $omega = 2\/3$ above, which is the
  /// classic optimum for a second-order operator on a regular grid and the
  /// weight a multigrid level wants rather than the undamped one.
  pub fn smoother(a: &CsrMatrix<T>) -> Self {
    Self::weighted(a, na::convert(2.0 / 3.0))
  }

  /// The weighted Jacobi inverse of an operator given by its diagonal alone.
  ///
  /// The diagonal is the whole datum, so an operator that never forms its
  /// entries still has one to offer: the diagonal of a sum is the sum of the
  /// diagonals, which a matrix-free operator can gather from its parts.
  pub fn from_diagonal(diag: &Vector<T>, omega: T::RealField) -> Self {
    assert!(
      diag.iter().all(|d| !d.is_zero()),
      "Jacobi needs a nonzero diagonal"
    );
    let omega = T::from_real(omega);
    Self {
      inv_diag: diag.map(|d| omega / d),
    }
  }
}

impl<T: Field> ApproxInverse for Jacobi<T> {
  type Space = Vector<T>;
  fn dim(&self) -> usize {
    self.inv_diag.len()
  }
  fn apply(&self, r: &Vector<T>) -> Vector<T> {
    r.component_mul(&self.inv_diag)
  }
}

impl<T: Field> SelfAdjoint for Jacobi<T> {}

/// A block-diagonal approximate inverse $B = "diag"(B_0, dots, B_(m-1))$: apply
/// each block's inverse to the corresponding contiguous slice of the vector.
///
/// The natural preconditioner for a saddle-point system, where the theory
/// (operator preconditioning) prescribes a norm that is block-diagonal across
/// the spaces. Self-adjoint exactly when every block is, so it may precondition
/// [`cg`](crate::krylov::cg) / [`minres`](crate::krylov::minres) precisely when
/// the blocks do.
///
/// The blocks share one type, which is what keeps the apply monomorphized: for
/// the mixed Hodge-Laplace system every block is the same direct SPD solve of a
/// different Gram matrix.
#[derive(Clone, Debug)]
pub struct BlockDiagonal<B> {
  blocks: Vec<B>,
}

impl<B> BlockDiagonal<B> {
  /// The block inverses, in the order their spaces are stacked in the vector.
  pub fn new(blocks: Vec<B>) -> Self {
    Self { blocks }
  }
}

impl<T: Field, B: ApproxInverse<Space = Vector<T>>> ApproxInverse for BlockDiagonal<B> {
  type Space = Vector<T>;
  fn dim(&self) -> usize {
    self.blocks.iter().map(ApproxInverse::dim).sum()
  }
  fn apply(&self, r: &Vector<T>) -> Vector<T> {
    let mut out = Vector::zeros(self.dim());
    let mut offset = 0;
    for block in &self.blocks {
      let d = block.dim();
      let piece = block.apply(&r.rows(offset, d).into_owned());
      out.rows_mut(offset, d).copy_from(&piece);
      offset += d;
    }
    out
  }
}

impl<T: Field, B: SelfAdjoint<Space = Vector<T>>> SelfAdjoint for BlockDiagonal<B> {}
