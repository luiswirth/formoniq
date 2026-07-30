//! The three index families dispatched as one value.

use crate::{
  Sign,
  cartesian::{Word, WordDeletions, Words},
  monotone::{MonoDeletions, MonoIndex, MonoIndices, Symbols},
};

/// A basis element of one of the three index families.
///
/// The three are the bases of the three symmetry types:
/// [`Combination`](crate::Combination) a subset for $Lambda^k$,
/// [`Composition`](crate::Composition) an exponent vector for $"Sym"^k$, and
/// [`Word`] a word for $V^(times.circle k)$.
///
/// [`MonoIndex`] covers the two monotone ones, subsets and multisets, which the
/// shift makes a single bitset. [`Word`] covers the free one, where there
/// is no symmetry to quotient by and therefore nothing to compress: the two
/// representations differ because the objects do.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MultiIndex {
  /// A subset or a multiset: the basis of $Lambda^k$ or $"Sym"^k$.
  Mono(MonoIndex),
  /// A word: the basis of the free power $V^(times.circle k)$.
  Word(Word),
}

impl MultiIndex {
  pub fn degree(&self) -> usize {
    match self {
      Self::Mono(index) => index.degree(),
      Self::Word(index) => index.degree(),
    }
  }
  /// The rank in the family's own order: colex for the monotone ones, radix
  /// for the free one.
  pub fn rank(&self) -> usize {
    match self {
      Self::Mono(index) => index.rank(),
      Self::Word(index) => index.rank(),
    }
  }
  /// The symbols, in position order.
  pub fn word(&self) -> Symbols {
    match self {
      Self::Mono(index) => index.word(),
      Self::Word(index) => index.symbols(),
    }
  }
  pub fn symbol(&self, position: usize) -> usize {
    match self {
      Self::Mono(index) => index.symbol(position),
      Self::Word(index) => index.symbol(position),
    }
  }
  /// How many positions carry a symbol.
  pub fn multiplicity(&self, symbol: usize) -> usize {
    match self {
      Self::Mono(index) => index.multiplicity(symbol),
      Self::Word(index) => index.multiplicity(symbol),
    }
  }

  /// The monotone index, if this is one.
  pub fn as_mono(&self) -> Option<&MonoIndex> {
    match self {
      Self::Mono(index) => Some(index),
      Self::Word(_) => None,
    }
  }
  /// The cartesian index, if this is one.
  pub fn as_word(&self) -> Option<&Word> {
    match self {
      Self::Word(index) => Some(index),
      Self::Mono(_) => None,
    }
  }

  /// Combine two indices of one family: the signed merge on the quotients,
  /// concatenation on the free power.
  ///
  /// `None` only where the result is the zero of the algebra, which happens on
  /// an alternating index alone. The symmetric merge and the free
  /// concatenation are both total.
  ///
  /// # Panics
  /// If the families differ.
  pub fn merge(&self, other: &Self) -> Option<(Sign, Self)> {
    match (self, other) {
      (Self::Mono(left), Self::Mono(right)) => {
        left.merge(right).map(|(sign, index)| (sign, index.into()))
      }
      (Self::Word(left), Self::Word(right)) => Some((Sign::Pos, left.concat(right).into())),
      _ => panic!("a merge is within one family"),
    }
  }

  /// Every single-position deletion, with the sign it carries.
  ///
  /// Alternating deletions alternate in sign, symmetric and free ones do not.
  /// A free deletion is the simplest of the three: positions are distinct and
  /// nothing is reordered, so there is no permutation to take a sign from.
  pub fn deletions(&self) -> MultiDeletions {
    match self {
      Self::Mono(index) => MultiDeletions::Mono(index.deletions()),
      Self::Word(index) => MultiDeletions::Word(index.deletions()),
    }
  }
}

/// Every index of a family, degree and alphabet.
///
/// An enum rather than a boxed trait object: the monotone path allocates
/// nothing, and both variants are now the same size, so nothing is paid for
/// the family that is not in use.
#[derive(Debug, Clone)]
pub enum MultiIndices {
  Mono(MonoIndices),
  Word(Words),
}

impl Iterator for MultiIndices {
  type Item = MultiIndex;
  fn next(&mut self) -> Option<MultiIndex> {
    match self {
      Self::Mono(indices) => indices.next().map(MultiIndex::Mono),
      Self::Word(indices) => indices.next().map(MultiIndex::Word),
    }
  }
  fn size_hint(&self) -> (usize, Option<usize>) {
    match self {
      Self::Mono(indices) => indices.size_hint(),
      Self::Word(indices) => indices.size_hint(),
    }
  }
}
impl ExactSizeIterator for MultiIndices {}

/// Every single-position deletion of a [`MultiIndex`], with its sign.
#[derive(Debug, Clone)]
pub enum MultiDeletions {
  Mono(MonoDeletions),
  Word(WordDeletions),
}

impl Iterator for MultiDeletions {
  type Item = (Sign, usize, MultiIndex);
  fn next(&mut self) -> Option<Self::Item> {
    match self {
      Self::Mono(deletions) => deletions
        .next()
        .map(|(sign, symbol, reduced)| (sign, symbol, MultiIndex::Mono(reduced))),
      Self::Word(deletions) => deletions
        .next()
        .map(|(symbol, reduced)| (Sign::Pos, symbol, MultiIndex::Word(reduced))),
    }
  }
}

impl Default for MultiIndex {
  /// The empty alternating index: the basis of the scalars.
  fn default() -> Self {
    Self::Mono(MonoIndex::default())
  }
}

impl From<MonoIndex> for MultiIndex {
  fn from(index: MonoIndex) -> Self {
    Self::Mono(index)
  }
}
impl From<Word> for MultiIndex {
  fn from(index: Word) -> Self {
    Self::Word(index)
  }
}
