use super::{CanonicalVertplex, OrientedVertplex};
use crate::{combinatorics::Orientation, Dim, VertexIdx};

use itertools::Itertools as _;

/// A simplex consisting of explicitly ordered vertices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrderedVertplex(pub Vec<VertexIdx>);

/// constructors
impl OrderedVertplex {
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

impl OrderedVertplex {
  pub fn nvertices(&self) -> usize {
    self.0.len()
  }
  pub fn dim(&self) -> Dim {
    self.nvertices() - 1
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
  pub fn boundary(&self) -> Vec<OrientedVertplex> {
    self
      .subs(self.dim() - 1)
      .into_iter()
      .enumerate()
      .map(|(i, s)| {
        let orientation = Orientation::from_permutation_parity(self.nvertices() - 1 - i);
        OrientedVertplex::new(s, orientation)
      })
      .collect()
  }
}

impl std::ops::Index<usize> for OrderedVertplex {
  type Output = VertexIdx;
  fn index(&self, index: usize) -> &Self::Output {
    &self.0[index]
  }
}

// Equivalent conversions
impl From<Vec<VertexIdx>> for OrderedVertplex {
  fn from(value: Vec<VertexIdx>) -> Self {
    Self(value)
  }
}
impl From<OrderedVertplex> for Vec<VertexIdx> {
  fn from(value: OrderedVertplex) -> Self {
    value.0
  }
}
impl<const N: usize> From<[VertexIdx; N]> for OrderedVertplex {
  fn from(value: [VertexIdx; N]) -> Self {
    Self(value.to_vec())
  }
}
impl OrderedVertplex {
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

// Forgetful Conversion
impl OrderedVertplex {
  pub fn into_canonical(self) -> CanonicalVertplex {
    CanonicalVertplex::new(self.0)
  }
}
