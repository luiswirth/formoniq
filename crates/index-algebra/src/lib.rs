//! Combinatorics with sets of indicies

pub mod combinators;
pub mod sign;

use combinators::{
  IndexAntiBoundarySets, IndexBoundarySets, IndexPermutations, IndexSubsets, IndexSupsets,
};
use sign::{sort_signed, Sign};

pub fn binomial(n: usize, k: usize) -> usize {
  num_integer::binomial(n, k)
}
pub fn factorial(num: usize) -> usize {
  (1..=num).product()
}

// TODO: Sorted: seperate struct vs const generic bool vs enum vs runtime bool
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IndexSet {
  indices: Vec<usize>,
  is_sorted: bool,
}
impl Default for IndexSet {
  fn default() -> Self {
    Self {
      indices: Vec::new(),
      is_sorted: true,
    }
  }
}

impl IndexSet {
  pub fn new(indices: Vec<usize>) -> Self {
    let is_sorted = indices.is_sorted();
    Self { indices, is_sorted }
  }
  pub fn empty() -> Self {
    Self::default()
  }
  pub fn single(index: usize) -> Self {
    IndexSet::new(vec![index])
  }
  pub fn increasing(n: usize) -> Self {
    IndexSet::new((0..n).collect())
  }
}

impl IndexSet {
  pub fn indices(&self) -> &[usize] {
    &self.indices
  }
  pub fn is_sorted(&self) -> bool {
    self.is_sorted
  }
  pub fn len(&self) -> usize {
    self.indices.len()
  }
  pub fn is_empty(&self) -> bool {
    self.indices.is_empty()
  }
}

impl IndexSet {
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
impl From<Vec<usize>> for IndexSet {
  fn from(value: Vec<usize>) -> Self {
    Self::new(value)
  }
}
impl<const N: usize> From<[usize; N]> for IndexSet {
  fn from(value: [usize; N]) -> Self {
    Self::new(value.to_vec())
  }
}

impl IndexSet {
  pub fn signed(self, sign: impl Into<Sign>) -> SignedIndexSet {
    SignedIndexSet::new(self, sign)
  }
}

/// Lexicographical Ordering
impl IndexSet {
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
}

impl IndexSet {
  pub fn remove(&mut self, i: usize) -> usize {
    self.indices.remove(i)
  }

  pub fn subs(&self, ksub: usize) -> IndexSubsets {
    IndexSubsets::new(self.clone(), ksub)
  }
  pub fn sups(&self, len_sup: usize, base: IndexSet) -> IndexSupsets {
    IndexSupsets::new(self.clone(), len_sup, base)
  }
}

impl std::ops::Index<usize> for IndexSet {
  type Output = usize;
  fn index(&self, index: usize) -> &Self::Output {
    &self.indices[index]
  }
}

// Only Sorted
impl IndexSet {
  pub fn is_sub_of(&self, other: &IndexSet) -> bool {
    self.sub_cmp(other).map(|o| o.is_le()).unwrap_or(false)
  }
  pub fn is_sup_of(&self, other: &IndexSet) -> bool {
    self.sub_cmp(other).map(|o| o.is_ge()).unwrap_or(false)
  }
  /// Subset partial order relation.
  pub fn sub_cmp(&self, other: &IndexSet) -> Option<std::cmp::Ordering> {
    assert!(self.is_sorted && other.is_sorted);

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

impl IndexSet {
  pub fn union(self, mut other: IndexSet) -> IndexSet {
    let mut indices = self.indices;
    indices.append(&mut other.indices);
    IndexSet::new(indices)
  }
}

impl IndexSet {
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

    Self::new(indices)
  }

  pub fn from_graded_lex_rank(n: usize, k: usize, mut rank: usize) -> Self {
    rank -= Self::graded_lex_rank_offset(n, k);
    Self::from_lex_rank(n, k, rank)
  }

  pub fn lex_rank(&self, n: usize) -> usize {
    assert!(self.is_sorted);
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

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct SignedIndexSet {
  pub set: IndexSet,
  pub sign: Sign,
}

impl SignedIndexSet {
  pub fn new(set: impl Into<IndexSet>, sign: impl Into<Sign>) -> Self {
    Self {
      set: set.into(),
      sign: sign.into(),
    }
  }
  pub fn sign(&self) -> Sign {
    self.sign
  }

  /// Becomes empty if there is a duplicate index.
  pub fn sort(&mut self) {
    self.sign *= sort_signed(&mut self.set.indices);
    if self.set.indices.windows(2).any(|s| s[0] == s[1]) {
      self.set = IndexSet::empty();
    }
    self.set.is_sorted = true;
  }

  pub fn permutations(&self) -> IndexPermutations {
    IndexPermutations::new(self.clone())
  }

  pub fn boundary(&self) -> IndexBoundarySets {
    IndexBoundarySets::new(self.clone())
  }

  pub fn anti_boundary(&self, base: IndexSet) -> IndexAntiBoundarySets {
    IndexAntiBoundarySets::new(self.clone(), base)
  }
}
