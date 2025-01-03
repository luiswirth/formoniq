use crate::sign::Sign;

use super::{variants::*, IndexSet};

/// All signed permutations of `self` in lexicographical order.
///
/// It's lexicographical relative to the original set order.
/// It's only absolutly lexicographical, if the original set was sorted.
pub struct IndexPermutations<O: SetOrder, S: SetSign> {
  state: itertools::Permutations<std::vec::IntoIter<usize>>,
  set: IndexSet<O, S>,
}

impl<O: SetOrder, S: SetSign> IndexPermutations<O, S> {
  pub fn new(set: IndexSet<O, S>) -> Self {
    let k = set.len();
    Self::new_sub(set, k)
  }

  pub fn new_sub(set: IndexSet<O, S>, k: usize) -> Self {
    let indices = set.indices.clone().into_iter();
    let state = itertools::Itertools::permutations(indices, k);

    Self { state, set }
  }
}
impl IndexPermutations<CanonicalOrder, Unsigned> {
  pub fn canonical(n: usize) -> Self {
    Self::canonical_sub(n, n)
  }

  pub fn canonical_sub(n: usize, k: usize) -> Self {
    let set = IndexSet::increasing(n);
    Self::new_sub(set, k)
  }
}

impl<O: SetOrder, S: SetSign> Iterator for IndexPermutations<O, S> {
  type Item = IndexSet<ArbitraryOrder, Signed>;
  fn next(&mut self) -> Option<Self::Item> {
    let indices = self.state.next()?;
    let sorted = IndexSet::new(indices.clone())
      .with_sign(self.set.signedness.get_or_default())
      .try_sort_signed()
      .unwrap();
    let next = IndexSet {
      indices: indices.clone(),
      order: ArbitraryOrder,
      signedness: sorted.signedness,
    };

    Some(next)
  }
}

pub struct GradedIndexSubsets<O: SetOrder> {
  set: IndexSet<O, Unsigned>,
  k: usize,
}
impl<O: SetOrder> GradedIndexSubsets<O> {
  pub fn new<S: SetSign>(set: IndexSet<O, S>) -> Self {
    let set = set.forget_sign();
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
  pub fn new<S: SetSign>(set: IndexSet<O, S>, k: usize) -> Self {
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
  type Item = IndexSet<O, Unsigned>;
  fn next(&mut self) -> Option<Self::Item> {
    let indices = self.subsets.next()?;
    let next = IndexSet {
      indices,
      order: self.order,
      signedness: Unsigned,
    };
    Some(next)
  }
}

pub struct IndexBoundarySets<O: SetOrder, S: SetSign> {
  subsets: IndexSubsets<O>,
  signedness: S,
  boundary_sign: Sign,
}

impl<O: SetOrder, S: SetSign> IndexBoundarySets<O, S> {
  pub fn new(set: IndexSet<O, S>) -> Self {
    let k = set.len() - 1;
    let signedness = set.signedness;
    let subsets = IndexSubsets::new(set, k);
    let boundary_sign = Sign::from_parity(k);
    Self {
      subsets,
      signedness,
      boundary_sign,
    }
  }
}

impl<O: SetOrder, S: SetSign> Iterator for IndexBoundarySets<O, S> {
  type Item = IndexSet<O, Signed>;
  fn next(&mut self) -> Option<Self::Item> {
    let self_sign = self.signedness.get_or_default();
    let sign = self_sign * self.boundary_sign;
    self.boundary_sign.flip();

    let next = self.subsets.next()?;
    let next = IndexSet {
      indices: next.indices,
      order: next.order,
      signedness: Signed(sign),
    };
    Some(next)
  }
}

pub struct IndexSupsets {
  base_subsets: IndexSubsets<CanonicalOrder>,
  set: IndexSet<CanonicalOrder, Unsigned>,
}
impl IndexSupsets {
  pub fn new<S: SetSign>(
    set: IndexSet<CanonicalOrder, S>,
    universe: IndexSet<CanonicalOrder, S>,
    k: usize,
  ) -> Self {
    let base_set = IndexSet {
      indices: universe.indices,
      order: CanonicalOrder,
      signedness: Unsigned,
    };
    let base_subsets = IndexSubsets::new(base_set, k);
    let set = set.forget_sign();
    Self { base_subsets, set }
  }
}
impl Iterator for IndexSupsets {
  type Item = IndexSet<CanonicalOrder, Unsigned>;
  fn next(&mut self) -> Option<Self::Item> {
    let next = self.base_subsets.next()?;
    if self.set.is_sub_of(&next) {
      Some(next)
    } else {
      None
    }
  }
}

pub struct IndexAntiBoundarySets<S: SetSign> {
  supsets: IndexSupsets,
  signedness: S,
  boundary_sign: Sign,
}

impl<S: SetSign> IndexAntiBoundarySets<S> {
  pub fn new(set: IndexSet<CanonicalOrder, S>, universe: IndexSet<CanonicalOrder, S>) -> Self {
    let k = set.len() + 1;
    let signedness = set.signedness;
    let supsets = IndexSupsets::new(set, universe, k);
    let boundary_sign = Sign::from_parity(k);
    Self {
      supsets,
      signedness,
      boundary_sign,
    }
  }
}

impl<S: SetSign> Iterator for IndexAntiBoundarySets<S> {
  type Item = IndexSet<CanonicalOrder, Signed>;
  fn next(&mut self) -> Option<Self::Item> {
    let self_sign = self.signedness.get_or_default();
    let sign = self_sign * self.boundary_sign;
    self.boundary_sign.flip();

    let next = self.supsets.next()?;
    let next = IndexSet {
      indices: next.indices,
      order: next.order,
      signedness: Signed(sign),
    };
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
        assert!(a.lexicographical_cmp(b).is_lt());
      }
      for permut in permuts {
        let computed_sign = permut.sign();
        let expected_sign = permut.forget_sign().try_sort_signed().unwrap().sign();
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
