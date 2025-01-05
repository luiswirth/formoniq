//! Combinatorics with sets of indicies

#![allow(clippy::len_without_is_empty)]

pub mod combinators;
pub mod variants;

pub mod sign;

use combinators::{
  IndexAntiBoundarySets, IndexBoundarySets, IndexPermutations, IndexSubsets, IndexSupsets,
};
use variants::*;

use sign::Sign;

pub fn binomial(n: usize, k: usize) -> usize {
  num_integer::binomial(n, k)
}
pub fn factorial(num: usize) -> usize {
  (1..=num).product()
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct IndexSet<O: SetOrder, S: SetSign> {
  indices: Vec<usize>,
  order: O,
  signedness: S,
}

impl<O: SetOrder, S: SetSign> IndexSet<O, S> {
  pub fn indices(&self) -> &[usize] {
    &self.indices
  }
  pub fn signedness(&self) -> S {
    self.signedness
  }

  pub fn len(&self) -> usize {
    self.indices.len()
  }

  // Ignores differing k and only compares indicies lexicographically.
  pub fn pure_lexicographical_cmp(&self, other: &Self) -> std::cmp::Ordering {
    use std::cmp::Ordering as O;
    self
      .iter()
      .zip(other.iter())
      .find_map(|(a, b)| match a.cmp(b) {
        O::Equal => None,
        non_eq => Some(non_eq),
      })
      .unwrap_or(O::Equal)
  }
  // Compares indicies lexicographically, only when lengths are equal.
  pub fn partial_lexicographical_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
    if self.len() == other.len() {
      Some(self.pure_lexicographical_cmp(other))
    } else {
      None
    }
  }
  /// First compares indicies lexicographically, then the lengths.
  pub fn lexicographical_cmp(&self, other: &Self) -> std::cmp::Ordering {
    self
      .pure_lexicographical_cmp(other)
      .then(self.len().cmp(&other.len()))
  }
  /// First compares lengths, then indicies lexicographically.
  pub fn graded_lexicographical_cmp(&self, other: &Self) -> std::cmp::Ordering {
    self
      .len()
      .cmp(&other.len())
      .then_with(|| self.pure_lexicographical_cmp(other))
  }

  pub fn remove(&mut self, i: usize) -> usize {
    self.indices.remove(i)
  }

  pub fn permutations(&self) -> IndexPermutations<O, S> {
    IndexPermutations::new(self.clone())
  }

  pub fn subs(&self, ksub: usize) -> IndexSubsets<O> {
    IndexSubsets::new(self.clone(), ksub)
  }

  pub fn boundary(&self) -> IndexBoundarySets<O, S> {
    IndexBoundarySets::new(self.clone())
  }
}

impl<O: SetOrder, S: SetSign> std::ops::Index<usize> for IndexSet<O, S> {
  type Output = usize;
  fn index(&self, index: usize) -> &Self::Output {
    &self.indices[index]
  }
}

// Only Sorted
impl<S: SetSign> IndexSet<CanonicalOrder, S> {
  pub fn is_sub_of<S1: SetSign>(&self, other: &IndexSet<CanonicalOrder, S1>) -> bool {
    self.sub_cmp(other).map(|o| o.is_le()).unwrap_or(false)
  }
  pub fn is_sup_of<S1: SetSign>(&self, other: &IndexSet<CanonicalOrder, S1>) -> bool {
    self.sub_cmp(other).map(|o| o.is_ge()).unwrap_or(false)
  }
  /// Subset partial order relation.
  pub fn sub_cmp<S1: SetSign>(
    &self,
    other: &IndexSet<CanonicalOrder, S1>,
  ) -> Option<std::cmp::Ordering> {
    use std::cmp::Ordering as O;
    let mut is_le = true;
    let mut is_ge = true;

    let mut this = self.iter().peekable();
    let mut other = other.iter().peekable();
    while let (Some(self_v), Some(other_v)) = (this.peek(), other.peek()) {
      match self_v.cmp(other_v) {
        O::Equal => {
          this.next();
          other.next();
        }
        O::Less => {
          is_le = false;
          this.next();
        }
        O::Greater => {
          is_ge = false;
          other.next();
        }
      }
    }

    if this.next().is_some() {
      is_le = false;
    }
    if other.next().is_some() {
      is_ge = false;
    }

    match (is_le, is_ge) {
      (true, true) => Some(O::Equal),
      (true, false) => Some(O::Less),
      (false, true) => Some(O::Greater),
      _ => None,
    }
  }
}

/// Only Signed
impl<O: SetOrder> IndexSet<O, Signed> {
  pub fn sign(&self) -> Sign {
    self.signedness.0
  }
  pub fn sign_mut(&mut self) -> &mut Sign {
    &mut self.signedness.0
  }
}

/// Only Unsigned
impl<O: SetOrder> IndexSet<O, Unsigned> {
  pub fn union<O1: SetOrder>(
    self,
    mut other: IndexSet<O1, Unsigned>,
  ) -> IndexSet<ArbitraryOrder, Unsigned> {
    let mut indices = self.indices;
    indices.append(&mut other.indices);
    IndexSet {
      indices,
      order: ArbitraryOrder,
      signedness: Unsigned,
    }
  }
}

impl<S: SetSign> IndexSet<CanonicalOrder, S> {
  pub fn sups(&self, universe: Self, len_sup: usize) -> IndexSupsets {
    IndexSupsets::new(self.clone(), universe, len_sup)
  }

  pub fn anti_boundary(&self, universe: Self) -> IndexAntiBoundarySets<S> {
    IndexAntiBoundarySets::new(self.clone(), universe)
  }
}

/// Oriented == Sorted + Signed
impl IndexSet<ArbitraryOrder, Signed> {
  pub fn orientation_eq(&self, other: &Self) -> bool {
    self.indices == other.indices && self.sign() == other.sign()
  }
}

/// Only Local + Sorted + Unsigned
impl IndexSet<CanonicalOrder, Unsigned> {
  pub fn from_lex_rank(n: usize, k: usize, mut rank: usize) -> Self {
    let mut indices = Vec::with_capacity(k);
    let mut start = 0;
    for i in 0..k {
      let remaining = k - i;
      for x in start..=(n - remaining) {
        let c = binomial(n - x - 1, remaining - 1);
        if rank < c {
          indices.push(x);
          start = x + 1;
          break;
        } else {
          rank -= c;
        }
      }
    }

    Self {
      indices,
      order: CanonicalOrder,
      signedness: Unsigned,
    }
  }

  pub fn from_graded_lex_rank(n: usize, k: usize, mut rank: usize) -> Self {
    rank -= Self::graded_lex_rank_offset(n, k);
    Self::from_lex_rank(n, k, rank)
  }

  pub fn lex_rank(&self, n: usize) -> usize {
    let k = self.len();

    let mut rank = 0;
    for (i, &index) in self.iter().enumerate() {
      let start = if i == 0 { 0 } else { self[i - 1] + 1 };
      for s in start..index {
        rank += binomial(n - s - 1, k - i - 1);
      }
    }
    rank
  }

  pub fn graded_lex_rank(&self, n: usize) -> usize {
    let k = self.len();
    Self::graded_lex_rank_offset(n, k) + self.lex_rank(n)
  }

  fn graded_lex_rank_offset(n: usize, k: usize) -> usize {
    (0..k).map(|s| binomial(n, s)).sum()
  }
}

// Constructors
impl IndexSet<ArbitraryOrder, Unsigned> {
  pub fn new(indices: Vec<usize>) -> Self {
    Self {
      indices,
      ..Default::default()
    }
  }
}
impl IndexSet<CanonicalOrder, Unsigned> {
  pub fn none() -> Self {
    Self::default()
  }
  pub fn single(index: usize) -> Self {
    IndexSet::new(vec![index]).assume_sorted()
  }
}

impl IndexSet<CanonicalOrder, Unsigned> {
  pub fn increasing(n: usize) -> Self {
    IndexSet {
      indices: (0..n).collect(),
      order: CanonicalOrder,
      signedness: Unsigned,
    }
  }
}

// Conversions
impl<O: SetOrder, S: SetSign> IndexSet<O, S> {
  pub fn iter(&self) -> std::slice::Iter<usize> {
    self.indices.iter()
  }
  pub fn as_slice(&self) -> &[usize] {
    self.indices.as_slice()
  }
  pub fn into_vec(self) -> Vec<usize> {
    self.indices
  }
  pub fn into_array<const N: usize>(self) -> Result<[usize; N], Vec<usize>> {
    self.into_vec().try_into()
  }
}
impl From<Vec<usize>> for IndexSet<ArbitraryOrder, Unsigned> {
  fn from(value: Vec<usize>) -> Self {
    Self::new(value)
  }
}
impl<const N: usize> From<[usize; N]> for IndexSet<ArbitraryOrder, Unsigned> {
  fn from(value: [usize; N]) -> Self {
    Self::new(value.to_vec())
  }
}
