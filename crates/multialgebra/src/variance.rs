//! Whether a slot is built from the vector space or from its dual.

use gramian::{Gramian, Metric};

/// Whether a slot is built from $V$ (contravariant) or from $V^*$ (covariant).
///
/// The one datum with no representational footprint:
/// $dim Lambda^k (V) = dim Lambda^k (V^*)$, and likewise for every symmetric
/// factor, so the components of a multivector and a multiform are
/// indistinguishable. Nothing derives it and no shape check catches a wrong
/// one, so it is stated once at construction and propagated from there.
///
/// It decides the duality pairing, the direction of the functor (pushforward
/// against pullback), the musical isomorphisms and which Gramian measures a
/// slot. Never choose that Gramian by hand; go through [`Self::gramian`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Variance {
  /// Elements of $Lambda^k V$ and $"Sym"^k V$: vectors and multivectors.
  Contravariant,
  /// Elements of $Lambda^k V^*$ and $"Sym"^k V^*$: forms and multiforms.
  #[default]
  Covariant,
}

impl Variance {
  /// What this variance pairs against.
  pub fn dual(self) -> Self {
    match self {
      Self::Contravariant => Self::Covariant,
      Self::Covariant => Self::Contravariant,
    }
  }

  /// The Gramian measuring a slot of this variance: the metric tensor $g$ for
  /// the contravariant side, its inverse $g^(-1)$ for the covariant one.
  pub fn gramian(self, metric: &Metric) -> &Gramian {
    match self {
      Self::Contravariant => metric.vector_gramian(),
      Self::Covariant => metric.covector_gramian(),
    }
  }
}
