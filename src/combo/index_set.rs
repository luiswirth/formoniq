pub mod aliases;
pub mod variants;

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
  pub fn k(&self) -> usize {
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
    if self.k() == other.k() {
      Some(self.pure_lexicographical_cmp(other))
    } else {
      None
    }
  }
  /// First compares indicies lexicographically, then the lengths.
  pub fn lexicographical_cmp(&self, other: &Self) -> std::cmp::Ordering {
    self
      .pure_lexicographical_cmp(other)
      .then(self.k().cmp(&other.k()))
  }
  /// First compares lengths, then indicies lexicographically.
  pub fn graded_lexicographical_cmp(&self, other: &Self) -> std::cmp::Ordering {
    self
      .k()
      .cmp(&other.k())
      .then_with(|| self.pure_lexicographical_cmp(other))
  }

  /// All signed permutations of `self` in lexicographical order.
  pub fn permutations(&self) -> impl Iterator<Item = IndexSet<B, Ordered, Signed>> {
    let self_sign = self.signedness.get_or_default();
    let base = self.base.clone();

    // TODO: stop relying on implementation details of itertools
    itertools::Itertools::permutations(self.indices.clone().into_iter(), self.k())
      .enumerate()
      .map(move |(ipermut, indices)| {
        let permutation_sign = Sign::from_parity(ipermut);
        let sign = self_sign * permutation_sign;
        IndexSet {
          indices,
          base: base.clone(),
          order: Ordered,
          signedness: Signed(sign),
        }
      })
  }

  /// All unsigned subcombinations of length `ksub` in lexicographical order.
  pub fn subs(&self, ksub: usize) -> impl Iterator<Item = IndexSet<B, O, Unsigned>> {
    let base = self.base.clone();
    let ord = self.order;
    let indices = self.indices.clone().into_iter();
    // TODO: stop relying on implementation details of itertools
    itertools::Itertools::combinations(indices, ksub).map(move |indices| IndexSet {
      indices,
      base: base.clone(),
      order: ord,
      signedness: Unsigned,
    })
  }

  /// All signed boundary subcombinations.
  pub fn boundary(&self) -> impl Iterator<Item = IndexSet<B, O, Signed>> {
    let k = self.k();
    let self_sign = self.signedness.get_or_default();
    self.subs(k - 1).enumerate().map(move |(i, sub)| {
      let boundary_sign = Sign::from_parity(k - 1 - i);
      let sign = boundary_sign * self_sign;
      sub.with_sign(sign)
    })
  }
}

impl<B: Base, O: Order, S: Signedness> std::ops::Index<usize> for IndexSet<B, O, S> {
  type Output = usize;
  fn index(&self, index: usize) -> &Self::Output {
    &self.indices[index]
  }
}

/// Only Base
impl<B: Specified, O: Order, S: Signedness> IndexSet<B, O, S> {
  pub fn n(&self) -> usize {
    self.base.n()
  }
  pub fn base_indices(&self) -> Vec<usize> {
    self.base.indices()
  }
}

// Only Sorted
impl<B: Base, S: Signedness> IndexSet<B, Sorted, S> {
  pub fn is_subset_of<B1: Base, S1: Signedness>(&self, other: &IndexSet<B1, Sorted, S1>) -> bool {
    self.subset_cmp(other).map(|o| o.is_le()).unwrap_or(false)
  }
  pub fn is_superset_of<B1: Base, S1: Signedness>(&self, other: &IndexSet<B1, Sorted, S1>) -> bool {
    self.subset_cmp(other).map(|o| o.is_ge()).unwrap_or(false)
  }
  /// Subset partial order relation.
  pub fn subset_cmp<B1: Base, S1: Signedness>(
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
  pub fn sups(&self, ksup: usize) -> impl Iterator<Item = IndexSet<B, Sorted, Unsigned>> {
    let base = Self {
      indices: self.base.indices(),
      base: self.base.clone(),
      order: Sorted,
      signedness: self.signedness,
    };
    let this = self.clone();
    base.subs(ksup).filter(move |sup| this.is_subset_of(sup))
  }

  /// All signed anti-boundary supcombinations.
  pub fn anti_boundary(&self) -> impl Iterator<Item = IndexSet<B, Sorted, Signed>> {
    let k = self.k();
    let self_sign = self.signedness.get_or_default();
    self.sups(k + 1).enumerate().map(move |(i, sub)| {
      let boundary_sign = Sign::from_parity(k - 1 - i);
      let sign = boundary_sign * self_sign;
      sub.with_sign(sign)
    })
  }
}

/// Only Local + Sorted + Unsigned
impl IndexSet<Local, Sorted, Unsigned> {
  pub fn from_rank(n: usize, k: usize, mut rank: usize) -> Self {
    let nlast = n - 1;
    let klast = k - 1;

    let mut indices = Vec::with_capacity(k);
    let mut curr_idx = 0;
    for i in 0..k {
      loop {
        let binom = binomial(nlast - curr_idx, klast - i);
        if rank < binom {
          break;
        }
        rank -= binom;
        curr_idx += 1;
      }
      indices.push(curr_idx);
      curr_idx += 1;
    }

    Self {
      indices,
      base: Local(n),
      order: Sorted,
      signedness: Unsigned,
    }
  }

  pub fn rank(&self) -> usize {
    let n = self.n();
    let k = self.k();

    let mut rank = 0;
    let mut icurr = 0;
    for (i, &v) in self.iter().enumerate() {
      for j in icurr..v {
        rank += binomial(n - 1 - j, k - 1 - i);
      }
      icurr = v + 1;
    }
    rank
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
