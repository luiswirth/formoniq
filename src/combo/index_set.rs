//! Combinatorics with sets of indicies
//!
//! Consider turning this into it's own crate.
//! Possible names: indexalgebra, permutic

pub mod combinators;
pub mod variants;

use combinators::{
  IndexAntiBoundarySets, IndexBoundarySets, IndexPermutations, IndexSubsets, IndexSupsets,
};
use variants::*;

use super::{binomial, Sign};

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct IndexSet<B: Base, O: Order, S: Signedness> {
  indices: Vec<usize>,
  base: B,
  order: O,
  signedness: S,
}

impl<B: Base, O: Order, S: Signedness> IndexSet<B, O, S> {
  pub fn indices(&self) -> &[usize] {
    &self.indices
  }
  pub fn len(&self) -> usize {
    self.indices.len()
  }
  pub fn base(&self) -> &B {
    &self.base
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

  pub fn permutations(&self) -> IndexPermutations<B, O, S> {
    IndexPermutations::new(self.clone())
  }

  pub fn subs(&self, ksub: usize) -> IndexSubsets<B, O> {
    IndexSubsets::new(self.clone(), ksub)
  }

  pub fn boundary(&self) -> IndexBoundarySets<B, O, S> {
    IndexBoundarySets::new(self.clone())
  }
}

impl<B: Base, O: Order, S: Signedness> std::ops::Index<usize> for IndexSet<B, O, S> {
  type Output = usize;
  fn index(&self, index: usize) -> &Self::Output {
    &self.indices[index]
  }
}

// Only Sorted
impl<B: Base, S: Signedness> IndexSet<B, Sorted, S> {
  pub fn is_sub_of<B1: Base, S1: Signedness>(&self, other: &IndexSet<B1, Sorted, S1>) -> bool {
    self.sub_cmp(other).map(|o| o.is_le()).unwrap_or(false)
  }
  pub fn is_sup_of<B1: Base, S1: Signedness>(&self, other: &IndexSet<B1, Sorted, S1>) -> bool {
    self.sub_cmp(other).map(|o| o.is_ge()).unwrap_or(false)
  }
  /// Subset partial order relation.
  pub fn sub_cmp<B1: Base, S1: Signedness>(
    &self,
    other: &IndexSet<B1, Sorted, S1>,
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
impl<B: Base, O: Order> IndexSet<B, O, Signed> {
  pub fn sign(&self) -> Sign {
    self.signedness.0
  }
}

/// Only Base + Sorted
impl<B: Specified, S: Signedness> IndexSet<B, Sorted, S> {
  pub fn sups(&self, len_sup: usize) -> IndexSupsets<B> {
    IndexSupsets::new(self.clone(), len_sup)
  }

  pub fn anti_boundary(&self) -> IndexAntiBoundarySets<B, S> {
    IndexAntiBoundarySets::new(self.clone())
  }
}

/// Oriented == Sorted + Signed
impl<B: Base> IndexSet<B, Ordered, Signed> {
  pub fn orientation_eq(&self, other: &Self) -> bool {
    self.indices == other.indices && self.sign() == other.sign()
  }
}

/// Only Local + Sorted + Unsigned
impl IndexSet<Local, Sorted, Unsigned> {
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
      base: Local(n),
      order: Sorted,
      signedness: Unsigned,
    }
  }

  pub fn from_graded_lex_rank(n: usize, k: usize, mut rank: usize) -> Self {
    rank -= Self::graded_lex_rank_offset(n, k);
    Self::from_lex_rank(n, k, rank)
  }

  pub fn lex_rank(&self) -> usize {
    let n = self.base.len();
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

  pub fn graded_lex_rank(&self) -> usize {
    let n = self.base.len();
    let k = self.len();
    Self::graded_lex_rank_offset(n, k) + self.lex_rank()
  }

  fn graded_lex_rank_offset(n: usize, k: usize) -> usize {
    (0..k).map(|s| binomial(n, s)).sum()
  }
}

// Constructors
impl IndexSet<Unspecified, Ordered, Unsigned> {
  pub fn new(indices: Vec<usize>) -> Self {
    Self {
      indices,
      ..Default::default()
    }
  }
}
impl IndexSet<Unspecified, Sorted, Unsigned> {
  pub fn none() -> Self {
    Self::default()
  }
  pub fn single(index: usize) -> Self {
    IndexSet::new(vec![index]).assume_sorted()
  }
  pub fn counting(n: usize) -> Self {
    IndexSet::new((0..n).collect()).assume_sorted()
  }
}

impl IndexSet<Local, Sorted, Unsigned> {
  pub fn canonical_full(n: usize) -> Self {
    IndexSet {
      indices: (0..n).collect(),
      base: Local(n),
      order: Sorted,
      signedness: Unsigned,
    }
  }
}

// Conversions
impl<B: Base, O: Order, S: Signedness> IndexSet<B, O, S> {
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
impl From<Vec<usize>> for IndexSet<Unspecified, Ordered, Unsigned> {
  fn from(value: Vec<usize>) -> Self {
    Self::new(value)
  }
}
impl<const N: usize> From<[usize; N]> for IndexSet<Unspecified, Ordered, Unsigned> {
  fn from(value: [usize; N]) -> Self {
    Self::new(value.to_vec())
  }
}
