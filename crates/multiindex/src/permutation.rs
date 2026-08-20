//! Permutations: the symmetric group $S_n$ as a combinatorial index structure.

use crate::{Sign, factorial};

/// A permutation of ${0, dots, n-1}$, stored in one-line notation
/// $p = (p_0, dots, p_(n-1))$.
///
/// The third combinatorial object beside [`Combination`](crate::Combination)
/// (the subsets, basis of $Lambda^k$) and [`Composition`](crate::Composition)
/// (the exponent vectors, basis of $"Sym"^d$): the bijections, carrying the
/// sign homomorphism $"sgn": S_n -> {plus.minus 1}$.
///
/// Enumeration and [`rank`](Self::rank) are colexicographic, the crate-wide
/// convention: $p$ precedes $q$ iff the reversed word of $p$ precedes that of
/// $q$ lexicographically. Equivalently the last entry is the most significant.
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Permutation(Vec<usize>);

impl Permutation {
  /// The identity of $S_n$.
  pub fn identity(n: usize) -> Self {
    Self((0..n).collect())
  }

  /// From one-line notation.
  ///
  /// The hypothesis is that the word is a bijection of ${0, dots, n-1}$
  /// ([`Self::is_valid`]), and it is the caller's to hold. It usually holds by
  /// construction: an enumeration of $S_n$, a composition or an inverse of
  /// permutations already in hand, the sorting order of a word. Where the word
  /// comes from outside, [`Self::new_checked`] asks.
  ///
  /// A word that is not a bijection still builds, and every operation on it
  /// then returns a meaningless answer or panics on an out-of-range index.
  /// Nothing here is unsafe, and nothing silently reads out of bounds.
  pub fn new(parts: impl IntoIterator<Item = usize>) -> Self {
    Self(parts.into_iter().collect())
  }
  /// The permutation of that word, or `None` if it is not one: the constructor
  /// that verifies what [`Self::new`] takes on contract.
  pub fn new_checked(parts: impl IntoIterator<Item = usize>) -> Option<Self> {
    let this = Self::new(parts);
    this.is_valid().then_some(this)
  }
  /// Whether the stored word is a permutation at all: a bijection of
  /// ${0, dots, n-1}$, every entry in range and none repeating, which is
  /// exactly the contract [`Self::new`] takes on trust and
  /// [`Self::new_checked`] verifies.
  ///
  /// One pass and one scratch bitvector: injectivity on a finite set of its own
  /// cardinality is already surjectivity, so there is nothing further to check.
  pub fn is_valid(&self) -> bool {
    let mut seen = vec![false; self.0.len()];
    self
      .0
      .iter()
      .all(|&p| p < seen.len() && !std::mem::replace(&mut seen[p], true))
  }

  pub fn len(&self) -> usize {
    self.0.len()
  }
  pub fn is_empty(&self) -> bool {
    self.0.is_empty()
  }
  pub fn parts(&self) -> &[usize] {
    &self.0
  }
  pub fn into_parts(self) -> Vec<usize> {
    self.0
  }
  pub fn iter(&self) -> impl DoubleEndedIterator<Item = usize> + ExactSizeIterator + '_ {
    self.0.iter().copied()
  }

  /// The image $p(i)$.
  pub fn apply(&self, i: usize) -> usize {
    self.0[i]
  }

  /// The inverse permutation $p^(-1)$.
  pub fn inverse(&self) -> Self {
    let mut inv = vec![0; self.len()];
    for (i, &p) in self.0.iter().enumerate() {
      inv[p] = i;
    }
    Self(inv)
  }

  /// The composite $p compose q$, acting as $i |-> p(q(i))$.
  ///
  /// # Panics
  /// If the two permutations have different lengths.
  pub fn compose(&self, other: &Self) -> Self {
    assert_eq!(self.len(), other.len(), "composition needs equal lengths");
    Self(other.0.iter().map(|&i| self.0[i]).collect())
  }

  /// The number of inversions, pairs $i < j$ with $p_i > p_j$.
  pub fn ninversions(&self) -> usize {
    (0..self.len())
      .flat_map(|j| (0..j).map(move |i| (i, j)))
      .filter(|&(i, j)| self.0[i] > self.0[j])
      .count()
  }

  /// The sign $"sgn"(p) = (-1)^("inv"(p))$, the parity of the permutation.
  pub fn sign(&self) -> Sign {
    Sign::from_parity(self.ninversions())
  }

  /// Colexicographic rank among all permutations of the same length: the
  /// factorial number system $sum_j d_j dot j!$, where
  /// $d_j = \#{i < j : p_i < p_j}$ counts the smaller entries to the left of
  /// position $j$.
  ///
  /// Independent of the length, exactly as
  /// [`Combination::rank`](crate::Combination::rank) is independent of the
  /// ambient dimension: $d_j$ reads only positions $<= j$. So under the
  /// embedding $S_n arrow.r.hook S_(n+1)$ that raises every value by one and
  /// appends $0$, the rank is unchanged, and $S_n$ is an initial segment of
  /// [`Self::all`] at $n+1$.
  pub fn rank(&self) -> usize {
    (0..self.len())
      .map(|j| (0..j).filter(|&i| self.0[i] < self.0[j]).count() * factorial(j))
      .sum()
  }

  /// Inverse of [`Self::rank`]: the permutation of $S_n$ at the given
  /// colexicographic position.
  ///
  /// # Panics
  /// If `rank` is not below $n!$.
  pub fn from_rank(n: usize, rank: usize) -> Self {
    assert!(rank < factorial(n), "rank out of range");
    // Recover the digits $d_j$ from the most significant end, then place each
    // value as the $d_j$-th smallest still unused.
    let mut rank = rank;
    let mut digits = vec![0; n];
    for j in (0..n).rev() {
      let f = factorial(j);
      digits[j] = rank / f;
      rank %= f;
    }
    let mut available: Vec<usize> = (0..n).collect();
    let mut parts = vec![0; n];
    for j in (0..n).rev() {
      parts[j] = available.remove(digits[j]);
    }
    Self(parts)
  }

  /// All $n!$ permutations of ${0, dots, n-1}$ in colexicographic order.
  ///
  /// Total at the degenerate end: $S_0$ is the one empty permutation, not an
  /// empty enumeration.
  ///
  /// Colex on the word is lex on the reversed word, which is how this is
  /// generated: the standard lexicographic successor drives a reversed buffer.
  pub fn all(n: usize) -> impl Iterator<Item = Self> {
    // `word` runs through $S_n$ lexicographically. The emitted permutation is
    // its reversal.
    let mut word: Option<Vec<usize>> = Some((0..n).collect());
    std::iter::from_fn(move || {
      let current = word.as_ref()?.clone();
      word = lex_successor(current.clone());
      Some(Self(current.into_iter().rev().collect()))
    })
  }
}

/// The next word in lexicographic order, or `None` at the last one.
fn lex_successor(mut word: Vec<usize>) -> Option<Vec<usize>> {
  let pivot = (0..word.len().checked_sub(1)?).rfind(|&i| word[i] < word[i + 1])?;
  let successor = (pivot + 1..word.len())
    .rfind(|&i| word[i] > word[pivot])
    .expect("a pivot has a larger entry to its right");
  word.swap(pivot, successor);
  word[pivot + 1..].reverse();
  Some(word)
}

impl std::fmt::Debug for Permutation {
  fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
    f.debug_list().entries(self.0.iter()).finish()
  }
}

impl FromIterator<usize> for Permutation {
  fn from_iter<T: IntoIterator<Item = usize>>(iter: T) -> Self {
    Self::new(iter)
  }
}
