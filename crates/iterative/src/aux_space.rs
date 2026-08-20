//! The additive auxiliary-space preconditioner, one [`ApproxInverse`] built from
//! a smoother and a list of auxiliary corrections.
//!
//! A correction is a cheaper space $W_i$ carrying its own approximate inverse
//! $B_i$, tied to the main space by a transfer $Pi_i: W_i -> V$. It acts on a
//! residual by pulling it back, solving there, and pushing the result forward,
//! $r |-> Pi_i B_i Pi_i^T r$. The preconditioner is the additive (parallel)
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
//! coarsens in space along a mesh hierarchy, an auxiliary space coarsens in
//! structure onto a different discretization of the same problem, and the two
//! compose, each $B_i$ may itself be a V-cycle. This crate stays backend-free:
//! what the spaces $W_i$ and the transfers $Pi_i$ are is the consumer's
//! business, supplied as plain [`CsrMatrix`]es and boxed approximate inverses.
//!
//! Additive, not multiplicative: every piece reads the same residual $r$ and
//! their results are summed. That is what makes $B$ self-adjoint whenever its
//! pieces are (a multiplicative sweep would not be), hence a valid
//! [`cg`](crate::krylov::cg) preconditioner, and it is why the corrections carry
//! [`SelfAdjoint`] inverses rather than bare [`ApproxInverse`]s: an auxiliary
//! space of an SPD problem is preconditioned to precondition CG, and there is no
//! use here for a piece that would break that.

use crate::{ApproxInverse, CsrMatrix, Field, SelfAdjoint, Vector, adjoint};

/// One auxiliary correction: a space tied to the main one by a transfer, with an
/// approximate inverse of the operator restricted to it.
///
/// `prolong` is $Pi: W_"aux" -> V_"main"$, the inclusion of the
/// auxiliary space into the main one; `restrict` is $Pi^H$, cached at
/// construction. `inverse` is $B approx A_"aux"^(-1)$ on the auxiliary space,
/// self-adjoint so the correction $Pi B Pi^H$ is positive semidefinite.
struct Correction<T> {
  prolong: CsrMatrix<T>,
  restrict: CsrMatrix<T>,
  inverse: Box<dyn SelfAdjoint<Space = Vector<T>>>,
}

impl<T: Field> Correction<T> {
  fn apply(&self, r: &Vector<T>) -> Vector<T> {
    &self.prolong * self.inverse.apply(&(&self.restrict * r))
  }
}

/// An additive auxiliary-space preconditioner of the main-space operator.
///
/// Holds a smoother $S$ on the main space and any number of auxiliary
/// corrections, and applies their sum. With no corrections it degrades to the
/// smoother alone, the totality base case, an auxiliary-space preconditioner
/// of an empty auxiliary set being a plain smoother with no special-casing. The
/// smoother is kept generic (it is the same type applied every call), the
/// corrections boxed (they differ in type: a discrete-gradient block and a
/// vector-nodal block are not the same solver). The dispatch is off the assembly
/// hot path, one apply per Krylov step against matvec-dominated cost.
pub struct AuxiliarySpace<S, T = f64> {
  smoother: S,
  corrections: Vec<Correction<T>>,
}

impl<T: Field, S: SelfAdjoint<Space = Vector<T>>> AuxiliarySpace<S, T> {
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
  /// there. Its adjoint is the restriction.
  ///
  /// # Panics
  /// If `prolong` does not map into the main space (its row count must match the
  /// smoother's dimension) or out of the inverse's space (its column count must
  /// match the inverse's dimension).
  #[must_use]
  pub fn with_correction(
    mut self,
    prolong: CsrMatrix<T>,
    inverse: Box<dyn SelfAdjoint<Space = Vector<T>>>,
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
    let restrict = adjoint(&prolong);
    self.corrections.push(Correction {
      prolong,
      restrict,
      inverse,
    });
    self
  }
}

impl<T: Field, S: SelfAdjoint<Space = Vector<T>>> ApproxInverse for AuxiliarySpace<S, T> {
  type Space = Vector<T>;
  fn dim(&self) -> usize {
    self.smoother.dim()
  }
  fn apply(&self, r: &Vector<T>) -> Vector<T> {
    self
      .corrections
      .iter()
      .fold(self.smoother.apply(r), |acc, c| acc + c.apply(r))
  }
}

/// Self-adjoint whenever the smoother is: each correction $Pi B Pi^H$ is
/// symmetric (a congruence of the self-adjoint $B$) and positive semidefinite,
/// and a sum of self-adjoint operators is self-adjoint. Positive-definiteness is
/// the smoother's promise, the corrections only adding to it, exactly the
/// pattern the rest of the crate follows. It is what lets this preconditioner
/// drive [`cg`](crate::krylov::cg).
impl<T: Field, S: SelfAdjoint<Space = Vector<T>>> SelfAdjoint for AuxiliarySpace<S, T> {}
