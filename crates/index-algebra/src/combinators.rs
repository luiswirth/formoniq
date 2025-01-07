use crate::{sign::Sign, SignedIndexSet};

use super::{variants::*, IndexSet};

/// All signed permutations of `self` in lexicographical order.
///
/// It's lexicographical relative to the original set order.
/// It's only absolutly lexicographical, if the original set was sorted.
pub struct IndexSubPermutations {
  state: itertools::Permutations<std::vec::IntoIter<usize>>,
}

impl IndexSubPermutations {
  pub fn new<O: SetOrder>(set: IndexSet<O>, k: usize) -> Self {
    let indices = set.indices.clone().into_iter();
    let state = itertools::Itertools::permutations(indices, k);
    Self { state }
  }
}
impl IndexSubPermutations {
  pub fn canonical(n: usize, k: usize) -> Self {
    let set = IndexSet::increasing(n);
    Self::new(set, k)
  }
}

impl Iterator for IndexSubPermutations {
  type Item = IndexSet<ArbitraryOrder>;
  fn next(&mut self) -> Option<Self::Item> {
    let indices = self.state.next()?;
    let next = IndexSet::new(indices);
    Some(next)
  }
}

pub struct IndexPermutations {
  sub_permutations: IndexSubPermutations,
  set_sign: Sign,
}
impl IndexPermutations {
  pub fn new<O: SetOrder>(set: SignedIndexSet<O>) -> Self {
    let SignedIndexSet { set, sign } = set;
    let k = set.len();
    let sub_permutations = IndexSubPermutations::new(set, k);
    Self {
      sub_permutations,
      set_sign: sign,
    }
  }
  pub fn canonical(n: usize) -> Self {
    let set = IndexSet::increasing(n).with_sign(Sign::Pos);
    Self::new(set)
  }
}

impl Iterator for IndexPermutations {
  type Item = SignedIndexSet<ArbitraryOrder>;
  fn next(&mut self) -> Option<Self::Item> {
    let next = self.sub_permutations.next()?;
    let permut_sign = next.permut_sign().unwrap();
    let sign = self.set_sign * permut_sign;
    Some(next.with_sign(sign))
  }
}

pub struct GradedIndexSubsets<O: SetOrder> {
  set: IndexSet<O>,
  k: usize,
}
impl<O: SetOrder> GradedIndexSubsets<O> {
  pub fn new(set: IndexSet<O>) -> Self {
    let k = 0;
    Self { set, k }
  }
}
impl GradedIndexSubsets<CanonicalOrder> {
  pub fn canonical(n: usize) -> Self {
    let set = IndexSet::increasing(n);
    Self::new(set)
  }
}
impl<O: SetOrder> Iterator for GradedIndexSubsets<O> {
  type Item = IndexSubsets<O>;
  fn next(&mut self) -> Option<Self::Item> {
    (self.k <= self.set.len()).then(|| {
      let next = IndexSubsets::new(self.set.clone(), self.k);
      self.k += 1;
      next
    })
  }
}

pub struct IndexSubsets<O: SetOrder> {
  subsets: itertools::Combinations<std::vec::IntoIter<usize>>,
  order: O,
}

impl<O: SetOrder> IndexSubsets<O> {
  pub fn new(set: IndexSet<O>, k: usize) -> Self {
    let subsets = itertools::Itertools::combinations(set.indices.into_iter(), k);
    let order = set.order;
    Self { subsets, order }
  }
}
impl IndexSubsets<CanonicalOrder> {
  /// Sorted subsets of {1,...,n}
  pub fn canonical(n: usize, k: usize) -> Self {
    let set = IndexSet::increasing(n);
    Self::new(set, k)
  }
}

impl<O: SetOrder> Iterator for IndexSubsets<O> {
  type Item = IndexSet<O>;
  fn next(&mut self) -> Option<Self::Item> {
    let indices = self.subsets.next()?;
    let next = IndexSet {
      indices,
      order: self.order,
    };
    Some(next)
  }
}

pub struct IndexBoundarySets<O: SetOrder> {
  subsets: IndexSubsets<O>,
  sign: Sign,
}

impl<O: SetOrder> IndexBoundarySets<O> {
  pub fn new(set: SignedIndexSet<O>) -> Self {
    let k = set.set.len() - 1;
    let subsets = IndexSubsets::new(set.set, k);
    let boundary_sign = Sign::from_parity(k);
    let sign = set.sign * boundary_sign;
    Self { subsets, sign }
  }
}

impl<O: SetOrder> Iterator for IndexBoundarySets<O> {
  type Item = SignedIndexSet<O>;
  fn next(&mut self) -> Option<Self::Item> {
    let next = self.subsets.next()?.with_sign(self.sign);
    self.sign.flip();
    Some(next)
  }
}

pub struct IndexSupsets {
  root_subsets: IndexSubsets<CanonicalOrder>,
  set: IndexSet<CanonicalOrder>,
}
impl IndexSupsets {
  pub fn new(set: IndexSet<CanonicalOrder>, root: IndexSet<CanonicalOrder>, k: usize) -> Self {
    let root_subsets = IndexSubsets::new(root, k);
    Self { root_subsets, set }
  }
}
impl Iterator for IndexSupsets {
  type Item = IndexSet<CanonicalOrder>;
  fn next(&mut self) -> Option<Self::Item> {
    let next = self.root_subsets.next()?;
    self.set.is_sub_of(&next).then_some(next)
  }
}

pub struct IndexAntiBoundarySets {
  supsets: IndexSupsets,
  sign: Sign,
}

impl IndexAntiBoundarySets {
  pub fn new(set: SignedIndexSet<CanonicalOrder>, root: IndexSet<CanonicalOrder>) -> Self {
    let k = set.set.len() + 1;
    let supsets = IndexSupsets::new(set.set, root, k);
    let boundary_sign = Sign::from_parity(k);
    let sign = set.sign * boundary_sign;
    Self { supsets, sign }
  }
}

impl Iterator for IndexAntiBoundarySets {
  type Item = SignedIndexSet<CanonicalOrder>;
  fn next(&mut self) -> Option<Self::Item> {
    let next = self.supsets.next()?.with_sign(self.sign);
    self.sign.flip();
    Some(next)
  }
}

#[cfg(test)]
mod test {
  use crate::IndexSet;

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
        let computed_sign = permut.sign;
        let expected_sign = permut.set.clone().into_sorted_signed().sign;
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
