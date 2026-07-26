//! The additive auxiliary-space preconditioner, one [`ApproxInverse`] built from
//! a smoother and a list of *auxiliary corrections*.
//!
//! A correction is a cheaper space $W_i$ carrying its own approximate inverse
//! $B_i$, tied to the main space by a transfer $Pi_i: W_i -> V$; it acts on a
//! residual by pulling it back, solving there, and pushing the result forward,
//! $r |-> Pi_i B_i Pi_i^T r$. The preconditioner is the *additive* (parallel)
//! sum of a smoother $S$ on $V$ itself with every correction,
//!
//! $ B = S + sum_i Pi_i B_i Pi_i^T, $
//!
//! the abstract form of the fictitious-space lemma (Nepomnyaschikh) and of
//! Hiptmair-Xu auxiliary-space preconditioning: the smoother damps the
//! high-frequency error the main space resolves, and each auxiliary space
//! handles a part of the near-kernel the smoother cannot see, moved onto a space
//! where a solver is effective.
//!
//! It is the natural counterpart of [`VCycle`](crate::VCycle): a V-cycle
//! coarsens in *space* along a mesh hierarchy, an auxiliary space coarsens in
//! *structure* onto a different discretization of the same problem, and the two
//! compose --- each $B_i$ may itself be a V-cycle. This crate stays backend-free:
//! what the spaces $W_i$ and the transfers $Pi_i$ *are* is the consumer's
//! business, supplied as plain [`CsrMatrix`]es and boxed approximate inverses.
//!
//! Additive, not multiplicative: every piece reads the same residual $r$ and
//! their results are summed. That is what makes $B$ self-adjoint whenever its
//! pieces are (a multiplicative sweep would not be), hence a valid
//! [`cg`](crate::krylov::cg) preconditioner, and it is why the corrections carry
//! [`SelfAdjoint`] inverses rather than bare [`ApproxInverse`]s: an auxiliary
//! space of an SPD problem is preconditioned to precondition CG, and there is no
//! use here for a piece that would break that.

use crate::{ApproxInverse, CsrMatrix, SelfAdjoint, Vector};

/// One auxiliary correction: a space tied to the main one by a transfer, with an
/// approximate inverse of the operator restricted to it.
///
/// `prolong` is $Pi: RR^(n_"aux") -> RR^(n_"main")$, the inclusion of the
/// auxiliary space into the main one; `restrict` is $Pi^T$, cached at
/// construction. `inverse` is $B approx A_"aux"^(-1)$ on the auxiliary space,
/// self-adjoint so the correction $Pi B Pi^T$ is symmetric positive
/// semidefinite.
struct Correction {
  prolong: CsrMatrix,
  restrict: CsrMatrix,
  inverse: Box<dyn SelfAdjoint<Space = Vector>>,
}

impl Correction {
  fn apply(&self, r: &Vector) -> Vector {
    &self.prolong * self.inverse.apply(&(&self.restrict * r))
  }
}

/// An additive auxiliary-space preconditioner of the main-space operator.
///
/// Holds a smoother $S$ on the main space and any number of auxiliary
/// corrections, and applies their sum. With no corrections it degrades to the
/// smoother alone --- the totality base case, an auxiliary-space preconditioner
/// of an empty auxiliary set being a plain smoother with no special-casing. The
/// smoother is kept generic (it is the same type applied every call), the
/// corrections boxed (they differ in type: a discrete-gradient block and a
/// vector-nodal block are not the same solver); the dispatch is off the assembly
/// hot path, one apply per Krylov step against matvec-dominated cost.
pub struct AuxiliarySpace<S> {
  smoother: S,
  corrections: Vec<Correction>,
}

impl<S: SelfAdjoint> AuxiliarySpace<S> {
  /// A preconditioner from a smoother alone, corrections added by
  /// [`with_correction`](Self::with_correction).
  pub fn new(smoother: S) -> Self {
    Self {
      smoother,
      corrections: Vec::new(),
    }
  }

  /// Add an auxiliary correction: the transfer $Pi$ from the auxiliary space
  /// into the main one, and a self-adjoint approximate inverse of the operator
  /// there. Its transpose is the restriction.
  ///
  /// # Panics
  /// If `prolong` does not map into the main space (its row count must match the
  /// smoother's dimension) or out of the inverse's space (its column count must
  /// match the inverse's dimension).
  #[must_use]
  pub fn with_correction(
    mut self,
    prolong: CsrMatrix,
    inverse: Box<dyn SelfAdjoint<Space = Vector>>,
  ) -> Self {
    assert_eq!(
      prolong.nrows(),
      self.smoother.dim(),
      "prolongation must map into the main space"
    );
    assert_eq!(
      prolong.ncols(),
      inverse.dim(),
      "prolongation must map out of the auxiliary space"
    );
    let restrict = prolong.transpose();
    self.corrections.push(Correction {
      prolong,
      restrict,
      inverse,
    });
    self
  }
}

impl<S: SelfAdjoint<Space = Vector>> ApproxInverse for AuxiliarySpace<S> {
  type Space = Vector;
  fn dim(&self) -> usize {
    self.smoother.dim()
  }
  fn apply(&self, r: &Vector) -> Vector {
    self
      .corrections
      .iter()
      .fold(self.smoother.apply(r), |acc, c| acc + c.apply(r))
  }
}

/// Self-adjoint whenever the smoother is: each correction $Pi B Pi^T$ is
/// symmetric (a congruence of the self-adjoint $B$) and positive semidefinite,
/// and a sum of self-adjoint operators is self-adjoint. Positive-definiteness is
/// the smoother's promise, the corrections only adding to it, exactly the
/// pattern the rest of the crate follows. It is what lets this preconditioner
/// drive [`cg`](crate::krylov::cg).
impl<S: SelfAdjoint<Space = Vector>> SelfAdjoint for AuxiliarySpace<S> {}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{Identity, Jacobi};
  use nalgebra::DMatrix;
  use nalgebra_sparse::CooMatrix;

  /// The exact inverse of a fixed SPD matrix as a [`SelfAdjoint`] approximate
  /// inverse, for testing corrections against a known solve.
  struct DenseInverse(DMatrix<f64>);
  impl DenseInverse {
    fn spd(inv: DMatrix<f64>) -> Self {
      Self(inv.clone().try_inverse().unwrap())
    }
  }
  impl ApproxInverse for DenseInverse {
    type Space = Vector;
    fn dim(&self) -> usize {
      self.0.nrows()
    }
    fn apply(&self, r: &Vector) -> Vector {
      &self.0 * r
    }
  }
  impl SelfAdjoint for DenseInverse {}

  fn csr(m: &DMatrix<f64>) -> CsrMatrix {
    let mut coo = CooMatrix::new(m.nrows(), m.ncols());
    for i in 0..m.nrows() {
      for j in 0..m.ncols() {
        if m[(i, j)] != 0.0 {
          coo.push(i, j, m[(i, j)]);
        }
      }
    }
    CsrMatrix::from(&coo)
  }

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
      .with_correction(csr(&prolong), Box::new(DenseInverse::spd(m.clone())));

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
        Box::new(DenseInverse::spd(spd(4, 0.9))),
      )
      .with_correction(
        csr(&DMatrix::from_fn(n, 2, |i, j| ((i + 5 * j) as f64).cos())),
        Box::new(DenseInverse::spd(spd(2, 0.2))),
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
}
