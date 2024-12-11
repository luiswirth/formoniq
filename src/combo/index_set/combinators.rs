use crate::combo::Sign;

use super::{variants::*, IndexSet};

/// All signed permutations of `self` in lexicographical order.
///
/// It's lexicographical relative to the original set order.
/// It's only absolutly lexicographical, if the original set was sorted.
pub struct IndexPermutations<B: Base, O: Order, S: Signedness> {
  state: itertools::Permutations<std::vec::IntoIter<usize>>,
  set: IndexSet<B, O, S>,
}

impl<B: Base, O: Order, S: Signedness> IndexPermutations<B, O, S> {
  pub fn new(set: IndexSet<B, O, S>) -> Self {
    let k = set.len();
    let indices = set.indices.clone().into_iter();
    let state = itertools::Itertools::permutations(indices, k);

    Self { state, set }
  }
}
impl IndexPermutations<Local, Sorted, Unsigned> {
  pub fn canonical(n: usize) -> Self {
    let set = IndexSet::canonical_full(n);
    Self::new(set)
  }
}

impl<B: Base, O: Order, S: Signedness> Iterator for IndexPermutations<B, O, S> {
  type Item = IndexSet<B, Ordered, Signed>;
  fn next(&mut self) -> Option<Self::Item> {
    let indices = self.state.next()?;
    let sorted = IndexSet::new(indices.clone())
      .with_sign(self.set.signedness.get_or_default())
      .into_sorted();
    let next = IndexSet {
      indices: indices.clone(),
      base: self.set.base.clone(),
      order: Ordered,
      signedness: sorted.signedness,
    };

    Some(next)
  }
}

pub struct GradedIndexSubsets<B: Base, O: Order> {
  set: IndexSet<B, O, Unsigned>,
  k: usize,
}
impl<B: Base, O: Order> GradedIndexSubsets<B, O> {
  pub fn new<S: Signedness>(set: IndexSet<B, O, S>) -> Self {
    let set = set.forget_sign();
    let k = 0;
    Self { set, k }
  }
}
impl GradedIndexSubsets<Local, Sorted> {
  pub fn canonical(n: usize) -> Self {
    let set = IndexSet::canonical_full(n);
    Self::new(set)
  }
}
impl<B: Base, O: Order> Iterator for GradedIndexSubsets<B, O> {
  type Item = IndexSubsets<B, O>;
  fn next(&mut self) -> Option<Self::Item> {
    (self.k <= self.set.len()).then(|| {
      let next = IndexSubsets::new(self.set.clone(), self.k);
      self.k += 1;
      next
    })
  }
}

pub struct IndexSubsets<B: Base, O: Order> {
  subsets: itertools::Combinations<std::vec::IntoIter<usize>>,
  base: B,
  order: O,
}

impl<B: Base, O: Order> IndexSubsets<B, O> {
  pub fn new<S: Signedness>(set: IndexSet<B, O, S>, k: usize) -> Self {
    let subsets = itertools::Itertools::combinations(set.indices.into_iter(), k);
    let base = set.base;
    let order = set.order;
    Self {
      subsets,
      base,
      order,
    }
  }
}
impl IndexSubsets<Local, Sorted> {
  /// Sorted subsets of {1,...,n}
  pub fn canonical(n: usize, k: usize) -> Self {
    let set = IndexSet::canonical_full(n);
    Self::new(set, k)
  }
}

impl<B: Base, O: Order> Iterator for IndexSubsets<B, O> {
  type Item = IndexSet<B, O, Unsigned>;
  fn next(&mut self) -> Option<Self::Item> {
    let indices = self.subsets.next()?;
    let next = IndexSet {
      indices,
      base: self.base.clone(),
      order: self.order,
      signedness: Unsigned,
    };
    Some(next)
  }
}

pub struct IndexBoundarySets<B: Base, O: Order, S: Signedness> {
  subsets: IndexSubsets<B, O>,
  signedness: S,
  boundary_sign: Sign,
}

impl<B: Base, O: Order, S: Signedness> IndexBoundarySets<B, O, S> {
  pub fn new(set: IndexSet<B, O, S>) -> Self {
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

impl<B: Base, O: Order, S: Signedness> Iterator for IndexBoundarySets<B, O, S> {
  type Item = IndexSet<B, O, Signed>;
  fn next(&mut self) -> Option<Self::Item> {
    let self_sign = self.signedness.get_or_default();
    let sign = self_sign * self.boundary_sign;
    self.boundary_sign.flip();

    let next = self.subsets.next()?;
    let next = IndexSet {
      indices: next.indices,
      base: next.base,
      order: next.order,
      signedness: Signed(sign),
    };
    Some(next)
  }
}

pub struct IndexSupsets<B: Specified> {
  base_subsets: IndexSubsets<B, Sorted>,
  set: IndexSet<B, Sorted, Unsigned>,
}
impl<B: Specified> IndexSupsets<B> {
  pub fn new<S: Signedness>(set: IndexSet<B, Sorted, S>, k: usize) -> Self {
    let base_set = IndexSet {
      indices: set.base.indices(),
      base: set.base.clone(),
      order: Sorted,
      signedness: Unsigned,
    };
    let base_subsets = IndexSubsets::new(base_set, k);
    let set = set.forget_sign();
    Self { base_subsets, set }
  }
}
impl<B: Specified> Iterator for IndexSupsets<B> {
  type Item = IndexSet<B, Sorted, Unsigned>;
  fn next(&mut self) -> Option<Self::Item> {
    let next = self.base_subsets.next()?;
    if self.set.is_sub_of(&next) {
      Some(next)
    } else {
      None
    }
  }
}

pub struct IndexAntiBoundarySets<B: Specified, S: Signedness> {
  supsets: IndexSupsets<B>,
  signedness: S,
  boundary_sign: Sign,
}

impl<B: Specified, S: Signedness> IndexAntiBoundarySets<B, S> {
  pub fn new(set: IndexSet<B, Sorted, S>) -> Self {
    let k = set.len() + 1;
    let signedness = set.signedness;
    let supsets = IndexSupsets::new(set, k);
    let boundary_sign = Sign::from_parity(k);
    Self {
      supsets,
      signedness,
      boundary_sign,
    }
  }
}

impl<B: Specified, S: Signedness> Iterator for IndexAntiBoundarySets<B, S> {
  type Item = IndexSet<B, Sorted, Signed>;
  fn next(&mut self) -> Option<Self::Item> {
    let self_sign = self.signedness.get_or_default();
    let sign = self_sign * self.boundary_sign;
    self.boundary_sign.flip();

    let next = self.supsets.next()?;
    let next = IndexSet {
      indices: next.indices,
      base: next.base,
      order: next.order,
      signedness: Signed(sign),
    };
    Some(next)
  }
}

#[cfg(test)]
mod test {
  use super::{GradedIndexSubsets, IndexPermutations};
  use crate::combo::IndexSet;

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
        let expected_sign = permut.forget_sign().into_sorted().sign();
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
        assert_eq!(subset.graded_lex_rank(), rank);
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
