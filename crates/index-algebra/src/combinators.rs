use super::IndexSet;
use crate::{sign::Sign, SignedIndexSet};

/// All signed permutations of `self` in lexicographical order.
///
/// It's lexicographical relative to the original set order.
/// It's only absolutly lexicographical, if the original set was sorted.
pub struct IndexPermutations {
  state: itertools::Permutations<std::vec::IntoIter<usize>>,
  set: SignedIndexSet,
}

impl IndexPermutations {
  pub fn new(set: SignedIndexSet) -> Self {
    let k = set.set.len();
    Self::new_sub(set, k)
  }

  pub fn new_sub(set: SignedIndexSet, k: usize) -> Self {
    let indices = set.set.indices.clone().into_iter();
    let state = itertools::Itertools::permutations(indices, k);
    Self { state, set }
  }
}

impl IndexPermutations {
  pub fn canonical(n: usize) -> Self {
    Self::canonical_sub(n, n)
  }

  pub fn canonical_sub(n: usize, k: usize) -> Self {
    let set = IndexSet::increasing(n).signed(Sign::default());
    Self::new_sub(set, k)
  }
}

impl Iterator for IndexPermutations {
  type Item = SignedIndexSet;
  fn next(&mut self) -> Option<Self::Item> {
    let indices = self.state.next()?;

    let mut sorted = IndexSet::new(indices.clone()).signed(self.set.sign);
    sorted.sort();
    let sign = sorted.sign();

    let next = IndexSet::new(indices).signed(sign);
    Some(next)
  }
}

pub struct GradedIndexSubsets {
  set: IndexSet,
  k: usize,
}
impl GradedIndexSubsets {
  pub fn new(set: IndexSet) -> Self {
    let k = 0;
    Self { set, k }
  }
}
impl GradedIndexSubsets {
  pub fn canonical(n: usize) -> Self {
    let set = IndexSet::increasing(n);
    Self::new(set)
  }
}
impl Iterator for GradedIndexSubsets {
  type Item = IndexSubsets;
  fn next(&mut self) -> Option<Self::Item> {
    (self.k <= self.set.len()).then(|| {
      let next = IndexSubsets::new(self.set.clone(), self.k);
      self.k += 1;
      next
    })
  }
}

pub struct IndexSubsets {
  subsets: itertools::Combinations<std::vec::IntoIter<usize>>,
  is_sorted: bool,
}

impl IndexSubsets {
  pub fn new(set: IndexSet, k: usize) -> Self {
    let subsets = itertools::Itertools::combinations(set.indices.into_iter(), k);
    Self {
      subsets,
      is_sorted: set.is_sorted,
    }
  }

  /// Sorted subsets of {1,...,n}
  pub fn canonical(n: usize, k: usize) -> Self {
    let set = IndexSet::increasing(n);
    Self::new(set, k)
  }

  pub fn is_sorted(&self) -> bool {
    self.is_sorted
  }
}

impl Iterator for IndexSubsets {
  type Item = IndexSet;
  fn next(&mut self) -> Option<Self::Item> {
    let indices = self.subsets.next()?;
    let next = IndexSet::new(indices);
    Some(next)
  }
}

pub struct IndexBoundarySets {
  subsets: IndexSubsets,
  sign: Sign,
}

impl IndexBoundarySets {
  pub fn new(set: SignedIndexSet) -> Self {
    let k = set.set.len() - 1;
    let sign = set.sign * Sign::from_parity(k);
    let subsets = IndexSubsets::new(set.set, k);
    Self { subsets, sign }
  }
}

impl Iterator for IndexBoundarySets {
  type Item = SignedIndexSet;
  fn next(&mut self) -> Option<Self::Item> {
    let sign = self.sign;
    self.sign.flip();

    let next = self.subsets.next()?.signed(sign);
    Some(next)
  }
}

pub struct IndexSupsets {
  base_subsets: IndexSubsets,
  set: IndexSet,
}
impl IndexSupsets {
  pub fn new(set: IndexSet, k: usize, base: IndexSet) -> Self {
    assert!(set.is_sorted());
    assert!(base.is_sorted());
    let base_subsets = IndexSubsets::new(base, k);
    Self { base_subsets, set }
  }
}
impl Iterator for IndexSupsets {
  type Item = IndexSet;
  fn next(&mut self) -> Option<Self::Item> {
    let next = self.base_subsets.next()?;
    if self.set.is_sub_of(&next) {
      Some(next)
    } else {
      None
    }
  }
}

pub struct IndexAntiBoundarySets {
  supsets: IndexSupsets,
  sign: Sign,
}

impl IndexAntiBoundarySets {
  pub fn new(set: SignedIndexSet, base: IndexSet) -> Self {
    let k = set.set.len() + 1;
    let sign = set.sign * Sign::from_parity(k);
    let supsets = IndexSupsets::new(set.set, k, base);
    Self { supsets, sign }
  }
}

impl Iterator for IndexAntiBoundarySets {
  type Item = SignedIndexSet;
  fn next(&mut self) -> Option<Self::Item> {
    let sign = self.sign;
    self.sign.flip();

    let next = self.supsets.next()?.signed(sign);
    Some(next)
  }
}

#[cfg(test)]
mod test {
  use crate::{sign::Sign, IndexSet, SignedIndexSet};

  use super::{GradedIndexSubsets, IndexPermutations};

  #[test]
  fn canonical_permutations() {
    for n in 0..=8 {
      let permuts: Vec<_> = IndexPermutations::canonical(n).collect();
      for win in permuts.windows(2) {
        let [a, b] = win else { unreachable!() };
        assert!(a.set.lexicographical_cmp(&b.set).is_lt());
      }
      for permut in permuts {
        let computed_sign = permut.sign();
        let expected_sign = {
          let mut set = SignedIndexSet::new(permut.set, Sign::default());
          set.sort();
          set.sign()
        };
        assert_eq!(computed_sign, expected_sign);
      }
    }
  }

  #[test]
  fn canonical_subsets() {
    for n in 0..=8 {
      let graded_subsets: Vec<Vec<_>> = GradedIndexSubsets::canonical(n)
        .map(|s| s.collect())
        .collect();

      for subsets in graded_subsets.iter() {
        for win in subsets.windows(2) {
          let [a, b] = win else { unreachable!() };
          assert!(a.lexicographical_cmp(b).is_lt());
        }
        for subset in subsets {
          assert!(subset.indices.is_sorted());
        }
      }
      let linearized: Vec<_> = graded_subsets
        .into_iter()
        .flat_map(|s| s.into_iter())
        .collect();
      for win in linearized.windows(2) {
        let [a, b] = win else { unreachable!() };
        assert!(a.graded_lexicographical_cmp(b).is_lt());
      }
      for (rank, subset) in linearized.iter().enumerate() {
        assert_eq!(subset.graded_lex_rank(n), rank);
        assert_eq!(
          IndexSet::from_graded_lex_rank(n, subset.len(), rank),
          *subset
        );
      }
    }
  }

  #[test]
  fn complex4() {
    let n = 4;
    let computed: Vec<Vec<_>> = GradedIndexSubsets::canonical(n)
      .map(|s| s.map(|s| s.indices).collect())
      .collect();

    let expected: [&[&[usize]]; 5] = [
      &[&[]],
      &[&[0], &[1], &[2], &[3]],
      &[&[0, 1], &[0, 2], &[0, 3], &[1, 2], &[1, 3], &[2, 3]],
      &[&[0, 1, 2], &[0, 1, 3], &[0, 2, 3], &[1, 2, 3]],
      &[&[0, 1, 2, 3]],
    ];

    assert_eq!(computed, expected);
  }
}
