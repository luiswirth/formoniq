use crate::combo::Sign;

use super::{variants::*, IndexSet};

/// All signed permutations of `self` in lexicographical order relative to
/// set order (not absolute).
pub struct IndexPermutations<B: Base, O: Order, S: Signedness> {
  state: itertools::Permutations<std::vec::IntoIter<usize>>,
  permutation_sign: Sign,
  base: B,
  /// Sorted <=> Absolutely lexicographical, not only relative.
  order: O,
  signedness: S,
}

impl<B: Base, O: Order, S: Signedness> IndexPermutations<B, O, S> {
  pub fn new(set: IndexSet<B, O, S>) -> Self {
    let k = set.k();
    let state = itertools::Itertools::permutations(set.indices.into_iter(), k);
    let permutation_sign = Sign::Pos;
    let base = set.base;
    let order = set.order;
    let signedness = set.signedness;

    Self {
      state,
      permutation_sign,
      base,
      order,
      signedness,
    }
  }

  pub fn base(&self) -> &B {
    &self.base
  }
  pub fn order(&self) -> O {
    self.order
  }
  pub fn signedness(&self) -> S {
    self.signedness
  }
}

impl<B: Base, O: Order, S: Signedness> Iterator for IndexPermutations<B, O, S> {
  type Item = IndexSet<B, Ordered, Signed>;
  fn next(&mut self) -> Option<Self::Item> {
    let self_sign = self.signedness.get_or_default();
    let sign = self_sign * self.permutation_sign;
    self.permutation_sign.flip();

    let indices = self.state.next()?;
    let next = IndexSet {
      indices,
      base: self.base.clone(),
      order: Ordered,
      signedness: Signed(sign),
    };
    Some(next)
  }
}

// How about graded on top?

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
    let k = set.k() - 1;
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

pub struct IndexSupsets<B: Specified, S: Signedness> {
  base_subsets: IndexSubsets<B, Sorted>,
  set: IndexSet<B, Sorted, S>,
}
impl<B: Specified, S: Signedness> IndexSupsets<B, S> {
  pub fn new(set: IndexSet<B, Sorted, S>, k: usize) -> Self {
    let base_set = IndexSet {
      indices: set.base.indices(),
      base: set.base.clone(),
      order: Sorted,
      signedness: Unsigned,
    };
    let base_subsets = IndexSubsets::new(base_set, k);
    Self { base_subsets, set }
  }
}
impl<B: Specified, S: Signedness> Iterator for IndexSupsets<B, S> {
  type Item = IndexSet<B, Sorted, Unsigned>;
  fn next(&mut self) -> Option<Self::Item> {
    let next = self.base_subsets.next()?;
    if self.set.is_subset_of(&next) {
      Some(next)
    } else {
      None
    }
  }
}

pub struct IndexAntiBoundarySets<B: Specified, S: Signedness> {
  supsets: IndexSupsets<B, S>,
  signedness: S,
  boundary_sign: Sign,
}

impl<B: Specified, S: Signedness> IndexAntiBoundarySets<B, S> {
  pub fn new(set: IndexSet<B, Sorted, S>) -> Self {
    let k = set.k() - 1;
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
