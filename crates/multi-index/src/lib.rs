//! Combinatorics with sets of indicies

#![allow(clippy::len_without_is_empty)]

pub mod combinators;
pub mod variants;

pub mod sign;

use std::marker::PhantomData;

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
pub struct MultiIndex<O: IndexKind> {
  indices: Vec<usize>,
  _order: PhantomData<O>,
}

impl<O: IndexKind> MultiIndex<O> {
  pub fn indices(&self) -> &[usize] {
    &self.indices
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
      .find_map(|(a, b)| match a.cmp(&b) {
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

  pub fn global_to_local_subset(&self, subset: &Self) -> MultiIndex<ArbitraryList> {
    let subset = subset
      .iter()
      .map(|iglobal| {
        self
          .iter()
          .position(|iother| iglobal == iother)
          .expect("Not a subset.")
      })
      .collect();
    MultiIndex::new(subset)
  }

  pub fn union<O1: IndexKind>(self, mut other: MultiIndex<O1>) -> MultiIndex<ArbitraryList> {
    let mut indices = self.indices;
    indices.append(&mut other.indices);
    MultiIndex::new(indices)
  }

  pub fn remove(&mut self, i: usize) -> usize {
    self.indices.remove(i)
  }

  pub fn subsets(&self, ksub: usize) -> IndexSubsets<O> {
    IndexSubsets::new(self.clone(), ksub)
  }

  pub fn boundary(&self) -> IndexBoundarySets<O> {
    IndexBoundarySets::new(self.clone().with_sign(Sign::Pos))
  }

  pub fn permutations(&self) -> IndexPermutations {
    self.clone().with_sign(Sign::Pos).permutations()
  }
}

impl<O: IndexKind> std::ops::Index<usize> for MultiIndex<O> {
  type Output = usize;
  fn index(&self, index: usize) -> &Self::Output {
    &self.indices[index]
  }
}

impl MultiIndex<IncreasingSet> {
  pub fn is_subset_of(&self, other: &Self) -> bool {
    self.sub_cmp(other).map(|o| o.is_le()).unwrap_or(false)
  }
  pub fn is_sup_of(&self, other: &Self) -> bool {
    self.sub_cmp(other).map(|o| o.is_ge()).unwrap_or(false)
  }
  /// Subset partial order relation.
  pub fn sub_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
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

impl MultiIndex<IncreasingSet> {
  pub fn supsets(&self, len_sup: usize, root: &Self) -> IndexSupsets {
    IndexSupsets::new(self.clone(), root.clone(), len_sup)
  }
  pub fn anti_boundary(&self, root: &MultiIndex<IncreasingSet>) -> IndexAntiBoundarySets {
    IndexAntiBoundarySets::new(self.clone().with_sign(Sign::Pos), root.clone())
  }
}

impl MultiIndex<IncreasingSet> {
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
      _order: PhantomData,
    }
  }

  pub fn from_graded_lex_rank(n: usize, k: usize, mut rank: usize) -> Self {
    rank -= Self::graded_lex_rank_offset(n, k);
    Self::from_lex_rank(n, k, rank)
  }

  pub fn lex_rank(&self, n: usize) -> usize {
    let k = self.len();

    let mut rank = 0;
    for (i, index) in self.iter().enumerate() {
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
impl MultiIndex<ArbitraryList> {
  pub fn new(indices: Vec<usize>) -> Self {
    Self {
      indices,
      ..Default::default()
    }
  }
}

impl MultiIndex<IncreasingSet> {
  pub fn none() -> Self {
    Self::default()
  }
  pub fn single(index: usize) -> Self {
    MultiIndex {
      indices: vec![index],
      _order: PhantomData,
    }
  }
  pub fn increasing(n: usize) -> Self {
    MultiIndex {
      indices: (0..n).collect(),
      _order: PhantomData,
    }
  }
}

impl<O: IndexKind> IntoIterator for MultiIndex<O> {
  type Item = usize;
  type IntoIter = std::vec::IntoIter<Self::Item>;
  fn into_iter(self) -> Self::IntoIter {
    self.indices.into_iter()
  }
}

// Conversions
impl<O: IndexKind> MultiIndex<O> {
  pub fn iter(&self) -> std::iter::Copied<std::slice::Iter<'_, usize>> {
    self.indices.iter().copied()
  }
  pub fn as_slice(&self) -> &[usize] {
    self.indices.as_slice()
  }
  pub fn into_vec(self) -> Vec<usize> {
    self.indices
  }
}
impl From<Vec<usize>> for MultiIndex<ArbitraryList> {
  fn from(value: Vec<usize>) -> Self {
    Self::new(value)
  }
}
impl<const N: usize> From<[usize; N]> for MultiIndex<ArbitraryList> {
  fn from(value: [usize; N]) -> Self {
    Self::new(value.to_vec())
  }
}

impl<const N: usize, O: IndexKind> TryFrom<MultiIndex<O>> for [usize; N] {
  type Error = MultiIndex<O>;
  fn try_from(value: MultiIndex<O>) -> Result<Self, Self::Error> {
    value.indices.try_into().map_err(|indices| MultiIndex {
      indices,
      ..Default::default()
    })
  }
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct SignedIndexSet<O: IndexKind> {
  pub set: MultiIndex<O>,
  pub sign: Sign,
}
impl<O: IndexKind> SignedIndexSet<O> {
  pub fn new(set: MultiIndex<O>, sign: Sign) -> Self {
    Self { set, sign }
  }

  pub fn into_parts(self) -> (MultiIndex<O>, Sign) {
    (self.set, self.sign)
  }

  pub fn permutations(&self) -> IndexPermutations {
    IndexPermutations::new(self.clone())
  }
  pub fn boundary(&self) -> IndexBoundarySets<O> {
    IndexBoundarySets::new(self.clone())
  }
}

impl SignedIndexSet<IncreasingSet> {
  pub fn anti_boundary(&self, root: &MultiIndex<IncreasingSet>) -> IndexAntiBoundarySets {
    IndexAntiBoundarySets::new(self.clone(), root.clone())
  }
}
