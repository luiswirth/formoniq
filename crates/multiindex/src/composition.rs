//! Weak compositions: the multi-indices of the symmetric algebra.
//!
//! A [`Composition`] is a tuple $k in NN_0^p$ with $sum_i k_i = d$: the
//! exponent vector of the monomial $x^k$, hence the canonical basis of
//! $"Sym"^d (RR^p)$, the degree-$d$ part of the polynomial algebra.
//!
//! This is the symmetric counterpart of [`Combination`](crate::Combination),
//! which indexes $Lambda^k$. The two are dual in structure, not variants of one
//! thing: a combination forbids repetition and carries a
//! [`Sign`](crate::Sign) under permutation, a composition mandates neither and
//! carries no sign. Compositions form a graded monoid under addition --
//! $x^k x^(k') = x^(k + k')$ -- where combinations instead carry the wedge,
//! which is partial and signed.
//!
//! Stars and bars bijects with the subsets of $d + p - 1$ slots in two
//! complementary readings, the bars giving the $(p-1)$-subsets and the stars
//! the $d$-subsets. Only the latter preserves colex, and both are proved as
//! theorems here rather than used as the representation. Neither is natural in
//! the ambient size: each absorbs the *degree* $d$, which is unbounded (a
//! refinement level, a polynomial order), into the *index count* of a
//! combination, which is bounded by a dimension. Enumerating compositions
//! directly is what keeps the degree free.

use crate::{Repetition, binomial};

/// A weak composition $k in NN_0^p$ of degree $d = sum_i k_i$: the exponent
/// vector of the monomial $x^k$, a basis element of $"Sym"^d (RR^p)$.
///
/// The degree is unbounded. Order among compositions of a fixed shape is
/// colexicographic on the [word](Composition::word), the crate's one indexing
/// convention, shared with [`Combination`](crate::Combination) and decided
/// between them by [`Repetition`].
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct Composition {
  /// The parts. Their sum is the degree; the length is the number of parts.
  parts: Vec<usize>,
}

impl Composition {
  pub fn new(parts: Vec<usize>) -> Self {
    Self { parts }
  }
  /// The zero composition of $p$ parts: the unit of the monoid, the monomial
  /// $1$.
  pub fn zero(nparts: usize) -> Self {
    Self::new(vec![0; nparts])
  }

  pub fn nparts(&self) -> usize {
    self.parts.len()
  }
  /// The degree $d = sum_i k_i$: the total degree of the monomial $x^k$.
  pub fn degree(&self) -> usize {
    self.parts.iter().sum()
  }
  pub fn parts(&self) -> &[usize] {
    &self.parts
  }
  pub fn into_parts(self) -> Vec<usize> {
    self.parts
  }

  /// The number of compositions of degree `degree` into `nparts` parts,
  /// $binom(d + p - 1, p - 1)$ -- equivalently $dim "Sym"^d (RR^p)$.
  ///
  /// Total at the degenerate corners: no parts admit only the empty
  /// composition of degree zero.
  pub fn count(nparts: usize, degree: usize) -> usize {
    if nparts == 0 {
      usize::from(degree == 0)
    } else {
      binomial(degree + nparts - 1, nparts - 1)
    }
  }

  /// The monotone word of this composition: each symbol repeated with the
  /// multiplicity of its part, ascending.
  ///
  /// The multiset reading of the exponent vector, and the shape it shares with
  /// [`Combination`](crate::Combination). Ordering is defined on it, so the two
  /// families are enumerated and ranked by one implementation.
  pub fn word(&self) -> Vec<usize> {
    self
      .parts
      .iter()
      .enumerate()
      .flat_map(|(symbol, &multiplicity)| std::iter::repeat_n(symbol, multiplicity))
      .collect()
  }

  /// The composition whose [`Composition::word`] is `word`: the multiplicity of
  /// each symbol.
  pub fn from_word(nparts: usize, word: &[usize]) -> Self {
    let mut parts = vec![0; nparts];
    for &symbol in word {
      parts[symbol] += 1;
    }
    Self::new(parts)
  }

  /// Every composition of degree `degree` into `nparts` parts, in the
  /// colexicographic order of their [words](Composition::word).
  ///
  /// The same convention [`Combination`](crate::Combination) uses, on the same
  /// object: the two differ only in whether a symbol may repeat, and
  /// [`Repetition`] is where that is decided. Colex earns
  /// its keep by making a rank independent of the alphabet, so adding parts
  /// leaves every existing composition where it was.
  pub fn all(nparts: usize, degree: usize) -> impl Iterator<Item = Composition> {
    Repetition::Allowed
      .words(nparts, degree)
      .map(move |word| Self::from_word(nparts, &word))
  }

  /// The position of this composition in [`Composition::all`], its canonical
  /// index. Inverse to [`Composition::from_rank`].
  ///
  /// $sum_i binom(w_i + i, i + 1)$ on the word: the combinatorial number
  /// system, which never mentions the number of parts.
  pub fn rank(&self) -> usize {
    Repetition::Allowed.rank(&self.word())
  }

  /// The composition of degree `degree` into `nparts` parts at position `rank`
  /// of [`Composition::all`]. Inverse to [`Composition::rank`].
  ///
  /// # Panics
  /// If `rank` is not below [`Composition::count`].
  pub fn from_rank(nparts: usize, degree: usize, rank: usize) -> Self {
    Self::from_word(
      nparts,
      &Repetition::Allowed.word_from_rank(nparts, degree, rank),
    )
  }
}

impl std::ops::Add for &Composition {
  type Output = Composition;
  /// Monomial multiplication $x^k x^(k') = x^(k + k')$: the graded monoid, of
  /// degree the sum of the degrees.
  ///
  /// # Panics
  /// If the shapes differ -- the two must be compositions into the same parts.
  fn add(self, other: &Composition) -> Composition {
    assert_eq!(
      self.nparts(),
      other.nparts(),
      "compositions add within a fixed number of parts"
    );
    Composition::new(
      self
        .parts
        .iter()
        .zip(&other.parts)
        .map(|(a, b)| a + b)
        .collect(),
    )
  }
}

impl FromIterator<usize> for Composition {
  fn from_iter<T: IntoIterator<Item = usize>>(iter: T) -> Self {
    Self::new(iter.into_iter().collect())
  }
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::combinations;

  /// The enumeration has the dimension of $"Sym"^d (RR^p)$, is duplicate-free,
  /// and every element has the declared shape.
  #[test]
  fn count_is_the_symmetric_power_dimension() {
    for nparts in 0..=5 {
      for degree in 0..=6 {
        let all: Vec<_> = Composition::all(nparts, degree).collect();
        assert_eq!(all.len(), Composition::count(nparts, degree));
        for composition in &all {
          assert_eq!(composition.nparts(), nparts);
          assert_eq!(composition.degree(), degree);
        }
        let mut unique = all.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), all.len());
      }
    }
  }

  /// Ranking is the position in the enumeration, and inverts it.
  #[test]
  fn rank_inverts_the_enumeration() {
    for nparts in 0..=5 {
      for degree in 0..=6 {
        for (i, composition) in Composition::all(nparts, degree).enumerate() {
          assert_eq!(composition.rank(), i);
          assert_eq!(Composition::from_rank(nparts, degree, i), composition);
        }
      }
    }
  }

  /// The enumeration is colexicographic on the word: compare the largest
  /// symbol first, and the smallest last.
  #[test]
  fn order_is_colexicographic_on_the_word() {
    let colex_key = |composition: &Composition| {
      let mut word = composition.word();
      word.reverse();
      word
    };
    for nparts in 0..=5 {
      for degree in 0..=6 {
        let all: Vec<_> = Composition::all(nparts, degree).collect();
        for pair in all.windows(2) {
          assert!(colex_key(&pair[0]) < colex_key(&pair[1]));
        }
      }
    }
  }

  /// A rank is independent of the number of parts: adding parts leaves every
  /// existing composition where it was.
  ///
  /// This is what colex is for, and it is what the previous
  /// reverse-lexicographic order did not have -- a word's position drifted
  /// upward as parts were appended, so widening the alphabet renumbered the
  /// basis. The formula makes it plain: the sum runs over the word and never
  /// mentions `nparts`.
  #[test]
  fn rank_does_not_depend_on_the_number_of_parts() {
    for degree in 0..=4 {
      for nparts in 1..=4 {
        for composition in Composition::all(nparts, degree) {
          for wider in nparts..=6 {
            let widened = Composition::from_word(wider, &composition.word());
            assert_eq!(widened.rank(), composition.rank());
          }
        }
      }
    }
  }

  /// Stars and bars: compositions of degree $d$ into $p$ parts biject with the
  /// $d$-subsets of $d + p - 1$, order for order, by the shift
  /// $w_i |-> w_i + i$ on the word.
  ///
  /// The *stars* are the subset here, not the bars. Both readings biject, and
  /// they are complementary, but only this one is order-preserving under the
  /// shared colex convention: a bar set has $p - 1$ elements, so its rank
  /// depends on the number of parts, while the word has $d$ and its rank does
  /// not.
  ///
  /// A theorem about the two index sets, not the way either is built -- which
  /// is what leaves the degree unbounded here while a combination's index
  /// count stays bounded.
  #[test]
  fn stars_and_bars_bijects_with_combinations() {
    for nparts in 1..=5 {
      for degree in 0..=6 {
        let slots = degree + nparts - 1;
        let via_stars: Vec<Composition> = combinations(slots, degree)
          .map(|star_set| {
            let word: Vec<usize> = star_set
              .iter()
              .enumerate()
              .map(|(position, symbol)| symbol - position)
              .collect();
            Composition::from_word(nparts, &word)
          })
          .collect();
        assert_eq!(
          via_stars,
          Composition::all(nparts, degree).collect::<Vec<_>>()
        );
      }
    }
  }

  /// The graded monoid: addition is associative, the zero composition is its
  /// unit, and degrees add.
  #[test]
  fn addition_is_a_graded_monoid() {
    for nparts in 0..=4 {
      let zero = Composition::zero(nparts);
      for a in Composition::all(nparts, 3) {
        assert_eq!(&a + &zero, a);
        assert_eq!(&zero + &a, a);
        for b in Composition::all(nparts, 2) {
          let sum = &a + &b;
          assert_eq!(sum.degree(), a.degree() + b.degree());
          for c in Composition::all(nparts, 1) {
            assert_eq!(&(&a + &b) + &c, &a + &(&b + &c));
          }
        }
      }
    }
  }

  /// The degree is genuinely unbounded: past the 64-index ceiling a
  /// [`Combination`](crate::Combination) imposes, which is exactly the bound
  /// stars and bars would have inherited.
  #[test]
  fn degree_is_unbounded() {
    for degree in [63, 64, 65, 256] {
      let all: Vec<_> = Composition::all(2, degree).collect();
      assert_eq!(all.len(), degree + 1);
      assert_eq!(all[0].parts(), &[degree, 0]);
      assert_eq!(all[degree].parts(), &[0, degree]);
    }
    assert_eq!(Composition::all(4, 100).count(), Composition::count(4, 100));
  }
}
