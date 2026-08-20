//! Integer simplicial cohomology of the complex.
//!
//! The cochain complex
//!
//! $dots.c ->^(dif^(k-1)) C^k ->^(dif^k) C^(k+1) ->^(dif^(k+1)) dots.c$
//!
//! is the chain complex of [`homology`](super::homology) with its arrows
//! reversed: $dif^k$ is the transpose of $partial_(k+1)$, one incidence read the
//! other way, exactly as [`chain`](super::chain) says of the two differentials.
//! So a cohomology class $H^k = ker dif^k slash "im" dif^(k-1)$ and a homology
//! class are the same subquotient of the same matrix, and both are
//! the shared `quotient_generators` routine.
//!
//! There is deliberately no cohomological Betti number. Over $ZZ$ the free
//! ranks of $H^k$ and $H_k$ agree by universal coefficients, so
//! [`Complex::betti_number`] is the rank of both and a second function would be
//! one datum in two places. Metric-free (invariant 5): cohomology is a function
//! of the incidence alone.
//!
//! The generators pair with the homology ones (Kronecker), and that pairing is
//! nonsingular over $QQ$: a cohomology generator measures the periods of the
//! cycles. It need not be unimodular here, because
//! `quotient_generators` returns a $QQ$-basis of the free part rather than a
//! $ZZ$-basis of the integral lattice.

use super::{
  chain::{Chain, Cochain},
  complex::Complex,
};
use crate::Dim;
use crate::linalg::exact::{IntegerMatrix, quotient_generators};

impl Complex {
  /// The integer coboundary $dif^k: C^k -> C^(k+1)$, as the transpose of
  /// $partial_(k+1)$.
  ///
  /// Total over every grade, inheriting the totality of
  /// [`Self::integral_boundary`]: off the range it is the map between zero
  /// modules.
  pub fn integral_coboundary(&self, grade: Dim) -> IntegerMatrix {
    self.integral_boundary(grade + 1).transpose()
  }

  /// Representative cocycles whose classes are a basis of the free part of
  /// $H^k (K; ZZ)$, one [`Cochain`] per Betti number $b_k$.
  ///
  /// Each generator is a k-cocycle, $dif^k z = 0$, exactly: the coefficients
  /// are integers and the incidence entries are $plus.minus 1$, so nothing here
  /// is closed only to a tolerance. Its class generates a $ZZ$-summand of
  /// $H^k$, and the $b_k$ classes are independent modulo coboundaries.
  ///
  /// The caveats of `quotient_generators` apply: representatives chosen by
  /// the elimination order, never minimizers, spanning the free part over $QQ$
  /// without necessarily generating its integral lattice.
  pub fn cohomology_generators(&self, grade: impl Into<Dim>) -> Vec<Cochain<i64>> {
    let grade = grade.into();
    quotient_generators(
      &self.integral_coboundary(grade),
      &self.integral_coboundary(grade - 1),
    )
    .into_iter()
    .map(|cocycle| Cochain::from_vec(grade, cocycle))
    .collect()
  }

  /// Representative cocycles of a basis of the free part of the relative
  /// cohomology $H^k (K, partial K; ZZ)$, one per
  /// [`relative_betti_number`](Self::relative_betti_number).
  ///
  /// The relative cochains *are* the cochains vanishing on $partial K$, so the
  /// relative complex is the cochain complex on the interior simplices
  /// (the interior selection) and a class is written
  /// back out as a full-length cochain by extension by zero. That embedding is
  /// the natural one, not a padding convention.
  ///
  /// These are the cocycles of the essential-boundary-condition de Rham
  /// complex, whose harmonic space they represent.
  pub fn relative_cohomology_generators(&self, grade: impl Into<Dim>) -> Vec<Cochain<i64>> {
    let grade = grade.into();
    let interior = self.interior_selection(grade);
    let outgoing = self
      .integral_coboundary(grade)
      .submatrix(&self.interior_selection(grade + 1), &interior);
    let incoming = self
      .integral_coboundary(grade - 1)
      .submatrix(&interior, &self.interior_selection(grade - 1));

    quotient_generators(&outgoing, &incoming)
      .into_iter()
      .map(|cocycle| Cochain::from_vec(grade, interior.scatter(&cocycle)))
      .collect()
  }
}

/// The Kronecker pairing matrix $P_(i j) = chevron.l z^i, z_j chevron.r$ of a set
/// of cochains against a set of chains.
pub fn kronecker_matrix(cocycles: &[Cochain<i64>], cycles: &[Chain<i64>]) -> Vec<Vec<i64>> {
  cocycles
    .iter()
    .map(|cocycle| {
      cycles
        .iter()
        .map(|cycle| super::chain::pairing(cocycle, cycle))
        .collect()
    })
    .collect()
}
