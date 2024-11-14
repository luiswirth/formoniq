use super::{sort_count_swaps, Orientation};
use crate::{Dim, VertexIdx};

use itertools::Itertools as _;

/// A simplex consisting of ordered(!) vertices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrderedSimplex(pub Vec<VertexIdx>);

/// constructors
impl OrderedSimplex {
  pub fn new(vertices: Vec<VertexIdx>) -> Self {
    Self(vertices)
  }
  pub fn vertex(v: VertexIdx) -> Self {
    Self(vec![v])
  }
  pub fn edge(a: VertexIdx, b: VertexIdx) -> Self {
    Self(vec![a, b])
  }
}
impl OrderedSimplex {
  pub fn nvertices(&self) -> usize {
    self.0.len()
  }
  pub fn dim(&self) -> Dim {
    self.nvertices() - 1
  }

  pub fn vertices(&self) -> &[VertexIdx] {
    &self.0
  }
  pub fn into_vertices(self) -> Vec<VertexIdx> {
    self.0
  }

  pub fn into_sorted(self) -> SortedSimplex {
    SortedSimplex::new(self.0)
  }

  /// Generates all `dim`-subsimplicies of `self`.
  /// The order is defined here!
  pub fn subs(&self, dim: Dim) -> Vec<Self> {
    // TODO: don't rely on internals of itertools for ordering -> use own implementation
    self
      .0
      .iter()
      .copied()
      .combinations(dim + 1)
      .map(Self::new)
      .collect()
  }

  /// Generates all oriented boundary simplicies of `self`.
  ///
  /// The order is inherited from [`Self::subs`].
  /// This order is exactly the reverse of the sum sign from the mathematical
  /// definition of the boundary operator (chain).
  pub fn boundary(&self) -> Vec<OrientedSimplex> {
    self
      .subs(self.dim() - 1)
      .into_iter()
      .enumerate()
      .map(|(i, s)| {
        let orientation = Orientation::from_permutation_parity(self.nvertices() - 1 - i);
        OrientedSimplex::new(s, orientation)
      })
      .collect()
  }
}
impl From<Vec<VertexIdx>> for OrderedSimplex {
  fn from(value: Vec<VertexIdx>) -> Self {
    Self(value)
  }
}
impl OrderedSimplex {
  pub fn iter(&self) -> std::slice::Iter<'_, VertexIdx> {
    self.0.iter()
  }
}
impl std::ops::Index<usize> for OrderedSimplex {
  type Output = VertexIdx;
  fn index(&self, index: usize) -> &Self::Output {
    &self.0[index]
  }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrientedSimplex {
  ordered: OrderedSimplex,
  orientation: Orientation,

  sorted: SortedSimplex,
  sort_orientation: Orientation,
}
impl OrientedSimplex {
  pub fn new(ordered: OrderedSimplex, orientation: Orientation) -> Self {
    let (sorted, nswaps) = SortedSimplex::new_nswaps(ordered.clone().into_vertices());
    let sort_orientation = Orientation::from_permutation_parity(nswaps);
    Self {
      ordered,
      sorted,
      orientation,
      sort_orientation,
    }
  }
  pub fn vertex(v: VertexIdx) -> Self {
    Self::new(OrderedSimplex::new(vec![v]), Orientation::Pos)
  }

  pub fn nvertices(&self) -> usize {
    self.ordered.nvertices()
  }
  pub fn dim(&self) -> Dim {
    self.ordered.dim()
  }

  pub fn orientation(&self) -> Orientation {
    self.orientation
  }
  pub fn ordered(&self) -> &OrderedSimplex {
    &self.ordered
  }
  pub fn into_ordered(self) -> OrderedSimplex {
    self.ordered
  }
  pub fn sorted(&self) -> &SortedSimplex {
    &self.sorted
  }
  pub fn into_sorted(self) -> SortedSimplex {
    self.sorted
  }
  pub fn vertices(&self) -> &[VertexIdx] {
    self.ordered.vertices()
  }
  pub fn into_vertices(self) -> Vec<VertexIdx> {
    self.ordered.into_vertices()
  }

  pub fn total_orientation(&self) -> Orientation {
    self.orientation * self.sort_orientation
  }

  pub fn orientation_eq(&self, other: &Self) -> Option<bool> {
    if self.sorted == other.sorted {
      Some(self.total_orientation() == other.total_orientation())
    } else {
      None
    }
  }

  /// Generates all oriented boundary simplicies of `self`.
  pub fn boundary(&self) -> Vec<OrientedSimplex> {
    self
      .ordered
      .boundary()
      .into_iter()
      .map(|mut s| {
        s.orientation *= self.orientation;
        s
      })
      .collect()
  }
}

impl OrientedSimplex {
  pub fn iter(&self) -> std::slice::Iter<'_, VertexIdx> {
    self.ordered.iter()
  }
}
impl std::ops::Index<usize> for OrientedSimplex {
  type Output = VertexIdx;
  fn index(&self, index: usize) -> &Self::Output {
    &self.ordered[index]
  }
}

/// A simplex with a canonical ordering (sorted by increasing vertex index) of the vertices
///
/// Allows for easy comparison of vertex sets.
/// Allows for subset relation comparison ([`PartialEq`] impl).
/// Allows for generating all super simplicies given the cells.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SortedSimplex(Vec<VertexIdx>);

/// constructors
impl SortedSimplex {
  pub fn new(mut vertices: Vec<VertexIdx>) -> Self {
    vertices.sort();
    Self(vertices)
  }
  pub fn new_nswaps(mut vertices: Vec<VertexIdx>) -> (Self, usize) {
    let nswaps = sort_count_swaps(&mut vertices);
    (Self(vertices), nswaps)
  }
  pub fn new_unchecked(vertices: Vec<VertexIdx>) -> Self {
    debug_assert!(vertices.is_sorted());
    Self(vertices)
  }
  pub fn vertex(v: VertexIdx) -> SortedSimplex {
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

impl SortedSimplex {
  pub fn nvertices(&self) -> usize {
    self.0.len()
  }
  pub fn dim(&self) -> Dim {
    self.nvertices() - 1
  }

  pub fn vertices(&self) -> &[VertexIdx] {
    &self.0
  }
  pub fn into_vertices(self) -> Vec<VertexIdx> {
    self.0
  }

  pub fn into_ordered(self) -> OrderedSimplex {
    OrderedSimplex::new(self.0)
  }

  /// Generates all sorted `dim`-subsimplicies of `self`.
  /// Order of subs inherited from [`OrderedSimplex::subs`].
  pub fn subs(&self, dim: Dim) -> Vec<Self> {
    self
      .clone()
      .into_ordered()
      .subs(dim)
      .into_iter()
      .map(|s| Self::new_unchecked(s.0))
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

  pub fn iter(&self) -> std::slice::Iter<'_, VertexIdx> {
    self.0.iter()
  }
}
impl From<Vec<VertexIdx>> for SortedSimplex {
  fn from(value: Vec<VertexIdx>) -> Self {
    Self::new(value)
  }
}
impl From<OrderedSimplex> for SortedSimplex {
  fn from(value: OrderedSimplex) -> Self {
    Self::new(value.0)
  }
}
impl From<OrientedSimplex> for SortedSimplex {
  fn from(value: OrientedSimplex) -> Self {
    Self::new(value.into_vertices())
  }
}

/// Efficent implementation of subsimplex/subset partial order relation.
impl PartialOrd for SortedSimplex {
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
