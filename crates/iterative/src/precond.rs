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
  type Space = Vector;
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
/// The cheapest non-trivial approximate inverse, and the archetypal smoother:
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
    Self::weighted(a, 1.0)
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
  pub fn weighted(a: &CsrMatrix, omega: f64) -> Self {
    let n = a.nrows();
    let mut diag = Vector::zeros(n);
    for (i, j, &v) in a.triplet_iter() {
      if i == j {
        diag[i] = v;
      }
    }
    Self::from_diagonal(&diag, omega)
  }

  /// The weighted Jacobi inverse of an operator given by its diagonal alone.
  ///
  /// The diagonal is the whole datum, so an operator that never forms its
  /// entries still has one to offer: the diagonal of a sum is the sum of the
  /// diagonals, which a matrix-free operator can gather from its parts.
  pub fn from_diagonal(diag: &Vector, omega: f64) -> Self {
    assert!(
      diag.iter().all(|&d| d != 0.0),
      "Jacobi needs a nonzero diagonal"
    );
    Self {
      inv_diag: diag.map(|d| omega / d),
    }
  }
}

impl ApproxInverse for Jacobi {
  type Space = Vector;
  fn dim(&self) -> usize {
    self.inv_diag.len()
  }
  fn apply(&self, r: &Vector) -> Vector {
    r.component_mul(&self.inv_diag)
  }
}

impl SelfAdjoint for Jacobi {}

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

impl<B: ApproxInverse<Space = Vector>> BlockDiagonal<B> {
  /// The block inverses, in the order their spaces are stacked in the vector.
  pub fn new(blocks: Vec<B>) -> Self {
    Self { blocks }
  }
}

impl<B: ApproxInverse<Space = Vector>> ApproxInverse for BlockDiagonal<B> {
  type Space = Vector;
  fn dim(&self) -> usize {
    self.blocks.iter().map(ApproxInverse::dim).sum()
  }
  fn apply(&self, r: &Vector) -> Vector {
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

impl<B: SelfAdjoint<Space = Vector>> SelfAdjoint for BlockDiagonal<B> {}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::testutil::{csr, dense_solve, symmetric_from_spectrum};
  use crate::{LinearOperator, StopCriterion, Vector, krylov::cg};

  use na::DMatrix;

  #[test]
  fn identity_applies_unchanged() {
    let id = Identity::new(3);
    let r = Vector::from_column_slice(&[2.0, -1.0, 7.0]);
    assert_eq!(id.apply(&r), r);
  }

  /// Totality at the degenerate boundary: order zero is a defined, trivial op.
  #[test]
  fn identity_is_total_at_zero() {
    let id = Identity::new(0);
    assert_eq!(id.apply(&Vector::zeros(0)), Vector::zeros(0));
  }

  /// On a diagonal operator Jacobi is the exact inverse: $B A = I$.
  #[test]
  fn jacobi_inverts_a_diagonal_operator() {
    let a = csr(&DMatrix::from_diagonal(&Vector::from_column_slice(&[
      2.0, 5.0, 0.25, 8.0,
    ])));
    let b = Jacobi::new(&a);
    let x = Vector::from_column_slice(&[1.0, -3.0, 4.0, 2.0]);
    assert!((b.apply(&a.apply(&x)) - &x).norm() < 1e-12);
  }

  /// The law the `SelfAdjoint` marker promises: $angle.l B r, s angle.r =
  /// angle.l r, B s angle.r$. Verified on a full (non-diagonal) SPD operator,
  /// whose diagonal Jacobi reads.
  #[test]
  fn jacobi_is_self_adjoint() {
    let a = csr(&symmetric_from_spectrum(&[1.0, 2.0, 4.0, 7.0, 9.0]));
    let b = Jacobi::new(&a);
    let r = Vector::from_column_slice(&[1.0, -2.0, 3.0, 0.5, -1.0]);
    let s = Vector::from_column_slice(&[4.0, 1.0, -1.0, 2.0, 3.0]);
    assert!((b.apply(&r).dot(&s) - r.dot(&b.apply(&s))).abs() < 1e-12);
  }

  /// A block-diagonal preconditioner applies each block's inverse to its own
  /// slice: for a block-diagonal operator with exact (Jacobi-on-diagonal)
  /// blocks it is the exact inverse, so preconditioned CG converges in one step.
  #[test]
  fn block_diagonal_of_exact_blocks_is_exact() {
    // A block-diagonal operator, each block a distinct diagonal matrix.
    let sizes = [3usize, 4, 2];
    let n: usize = sizes.iter().sum();
    let diag = DMatrix::from_diagonal(&Vector::from_fn(n, |i, _| 1.0 + (i % 6) as f64));
    let a = csr(&diag);

    let mut blocks = Vec::new();
    let mut off = 0;
    for &d in &sizes {
      let sub = csr(&diag.view((off, off), (d, d)).into_owned());
      blocks.push(Jacobi::new(&sub));
      off += d;
    }
    let precond = BlockDiagonal::new(blocks);
    assert_eq!(precond.dim(), n);

    let b = Vector::from_fn(n, |i, _| (i as f64 - 4.0).cos());
    let (x, report) = cg(&a, &precond, &b, StopCriterion::rtol(1e-12));
    assert!(
      report.converged && report.iters <= 1,
      "iters = {}",
      report.iters
    );
    assert!((x - dense_solve(&diag, &b)).norm() < 1e-9);
  }

  /// Block-diagonal is self-adjoint when its blocks are, so it may precondition
  /// CG/MINRES exactly when they may.
  #[test]
  fn block_diagonal_is_self_adjoint_from_blocks() {
    let a1 = csr(&symmetric_from_spectrum(&[1.0, 2.0, 4.0]));
    let a2 = csr(&symmetric_from_spectrum(&[3.0, 5.0]));
    let precond = BlockDiagonal::new(vec![Jacobi::new(&a1), Jacobi::new(&a2)]);
    let r = Vector::from_fn(5, |i, _| (i as f64 + 1.0).ln());
    let s = Vector::from_fn(5, |i, _| (2.0 * i as f64).cos());
    assert!((precond.apply(&r).dot(&s) - r.dot(&precond.apply(&s))).abs() < 1e-12);
  }
}
