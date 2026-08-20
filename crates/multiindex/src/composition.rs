//! Weak compositions: the multi-indices of the symmetric algebra.
//!
//! A [`Composition`] is a tuple $k in NN_0^p$ with $sum_i k_i = d$: the
//! exponent vector of the monomial $x^k$, hence the canonical basis of
//! $"Sym"^d (RR^p)$, the degree-$d$ part of the polynomial algebra.
//!
//! This is the symmetric counterpart of [`Combination`](crate::Combination),
//! which indexes $Lambda^k$. The two are the two values of
//! [`Repetition`]: a combination forbids repetition and so carries a
//! [`Sign`](crate::Sign) under permutation, a composition allows it and carries
//! none. Compositions form a graded monoid under addition,
//! $x^k x^(k') = x^(k + k')$, where combinations instead carry the wedge,
//! which is partial and signed.
//!
//! What differs is the representation, and only for a reason: an exponent vector
//! here, a bitset in [`MonoIndex`](crate::MonoIndex). The bitset bounds the
//! shifted alphabet, and the degree of a composition is unbounded (a refinement
//! level, a polynomial order), so the two coexist and agree, which is a theorem
//! of this module rather than an accident.
//!
//! Stars and bars bijects with the subsets of $d + p - 1$ slots in two
//! complementary readings, the bars giving the $(p-1)$-subsets and the stars
//! the $d$-subsets. Only the latter preserves colex, and both are proved as
//! theorems here rather than used as the representation. Neither is natural in
//! the ambient size: each absorbs the degree $d$, which is unbounded (a
//! refinement level, a polynomial order), into the index count of a
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
  /// The parts. Their sum is the degree. The length is the number of parts.
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
  /// $binom(d + p - 1, p - 1)$, equivalently $dim "Sym"^d (RR^p)$.
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
  /// If the shapes differ, the two must be compositions into the same parts.
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
