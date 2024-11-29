use super::{OrderedVertplex, OrientedVertplex};
use crate::{combinatorics::sort_count_swaps, Dim, VertexIdx};

use itertools::Itertools as _;

/// A combinatorial simplex with vertices in a canonical order (sorted by increasing [`VertexIdx`]).
///
/// Allows for easy comparison of vertex sets.
/// Allows for subset relation comparison ([`PartialEq`] impl).
/// Allows for generating all super simplicies given the cells.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CanonicalVertplex(Vec<VertexIdx>);

/// constructors
impl CanonicalVertplex {
  pub fn new(mut vertices: Vec<VertexIdx>) -> Self {
    vertices.sort();
    Self(vertices)
  }
  /// Additionaly returns number of swaps, necessary to obtain canonical
  /// permutation, from given permutation.
  pub fn new_nswaps(mut vertices: Vec<VertexIdx>) -> (Self, usize) {
    let nswaps = sort_count_swaps(&mut vertices);
    (Self(vertices), nswaps)
  }
  pub fn new_unchecked(vertices: Vec<VertexIdx>) -> Self {
    debug_assert!(vertices.is_sorted());
    Self(vertices)
  }
  pub fn vertex(v: VertexIdx) -> CanonicalVertplex {
    Self(vec![v])
  }
  pub fn edge(a: VertexIdx, b: VertexIdx) -> Self {
    if a < b {
      Self(vec![a, b])
    } else {
      Self(vec![b, a])
    }
  }
}

impl CanonicalVertplex {
  pub fn nvertices(&self) -> usize {
    self.0.len()
  }
  pub fn dim(&self) -> Dim {
    self.nvertices() - 1
  }

  /// Generates all sorted `dim`-subsimplicies.
  /// Order of subs inherited from [`OrderedVertplex::subs`].
  pub fn subs(&self, dim: Dim) -> Vec<Self> {
    self
      .clone()
      .into_ordered()
      .subs(dim)
      .into_iter()
      // subs of canonical are still canonical
      .map(|s| Self::new_unchecked(s.into_vec()))
      .collect()
  }

  /// Generates all sorted `dim`-supsimplicies of `self`.
  pub fn sups<'a>(&self, dim: Dim, cells: impl Iterator<Item = &'a Self>) -> Vec<Self> {
    cells
      .flat_map(move |c| c.subs(dim))
      .filter(move |s| self <= s)
      .unique()
      .collect()
  }
}

/// Efficent implementation of subsimplex/subset partial order relation.
impl PartialOrd for CanonicalVertplex {
  fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
    use std::cmp::Ordering as O;
    let mut is_le = true;
    let mut is_ge = true;

    let mut this = self.0.iter().peekable();
    let mut other = other.0.iter().peekable();
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

    // Check if we have remaining elements.
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

impl std::ops::Index<usize> for CanonicalVertplex {
  type Output = VertexIdx;
  fn index(&self, index: usize) -> &Self::Output {
    &self.0[index]
  }
}

// Equivalent conversions
impl From<Vec<VertexIdx>> for CanonicalVertplex {
  fn from(value: Vec<VertexIdx>) -> Self {
    Self::new(value)
  }
}
impl From<CanonicalVertplex> for Vec<VertexIdx> {
  fn from(value: CanonicalVertplex) -> Self {
    value.0
  }
}
impl<const N: usize> From<[VertexIdx; N]> for CanonicalVertplex {
  fn from(value: [VertexIdx; N]) -> Self {
    Self(value.to_vec())
  }
}
impl CanonicalVertplex {
  pub fn as_slice(&self) -> &[VertexIdx] {
    &self.0
  }
  pub fn iter(&self) -> std::slice::Iter<'_, VertexIdx> {
    self.0.iter()
  }

  pub fn into_vec(self) -> Vec<VertexIdx> {
    self.into()
  }
}

// Synthesizing conversion
// Missing information synthesized
impl CanonicalVertplex {
  pub fn into_ordered(self) -> OrderedVertplex {
    OrderedVertplex::new(self.0)
  }
  pub fn into_oriented(self) -> OrientedVertplex {
    OrientedVertplex::new_pos(self.into_ordered())
  }
}
