use super::{CanonicalVertplex, OrderedVertplex};
use crate::{combinatorics::Orientation, Dim, VertexIdx};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrientedVertplex {
  ordered: OrderedVertplex,
  canonical: CanonicalVertplex,

  superimposed_orient: Orientation,
  order_orient: Orientation,
}
impl OrientedVertplex {
  pub fn new(ordered: OrderedVertplex, superimposed_orient: Orientation) -> Self {
    let (canonical, nswaps) = CanonicalVertplex::new_nswaps(ordered.clone().into_vec());
    let order_orient = Orientation::from_permutation_parity(nswaps);
    Self {
      ordered,
      canonical,
      superimposed_orient,
      order_orient,
    }
  }
  pub fn new_pos(ordered: OrderedVertplex) -> Self {
    Self::new(ordered, Orientation::Pos)
  }
  pub fn vertex(v: VertexIdx) -> Self {
    Self::new(OrderedVertplex::new(vec![v]), Orientation::Pos)
  }
}

impl OrientedVertplex {
  pub fn nvertices(&self) -> usize {
    self.ordered.nvertices()
  }
  pub fn dim(&self) -> Dim {
    self.ordered.dim()
  }

  pub fn superimposed_orient(&self) -> Orientation {
    self.superimposed_orient
  }
  pub fn order_orient(&self) -> Orientation {
    self.order_orient
  }
  pub fn total_orientation(&self) -> Orientation {
    self.superimposed_orient * self.order_orient
  }

  pub fn orientation_eq(&self, other: &Self) -> Option<bool> {
    if self.canonical == other.canonical {
      Some(self.total_orientation() == other.total_orientation())
    } else {
      None
    }
  }

  /// Generates all oriented boundary simplicies of `self`.
  pub fn boundary(&self) -> Vec<OrientedVertplex> {
    self
      .ordered
      .boundary()
      .into_iter()
      .map(|mut s| {
        s.superimposed_orient *= self.superimposed_orient;
        s
      })
      .collect()
  }
}

impl std::ops::Index<usize> for OrientedVertplex {
  type Output = VertexIdx;
  fn index(&self, index: usize) -> &Self::Output {
    &self.ordered[index]
  }
}
impl OrientedVertplex {
  pub fn iter(&self) -> std::slice::Iter<'_, VertexIdx> {
    self.ordered.iter()
  }

  pub fn as_ordered(&self) -> &OrderedVertplex {
    &self.ordered
  }
  pub fn into_ordered(self) -> OrderedVertplex {
    self.ordered
  }
  pub fn as_canonical(&self) -> &CanonicalVertplex {
    &self.canonical
  }
  pub fn into_canonical(self) -> CanonicalVertplex {
    self.canonical
  }
  pub fn as_slice(&self) -> &[VertexIdx] {
    self.ordered.as_slice()
  }
  pub fn into_vec(self) -> Vec<VertexIdx> {
    self.ordered.into_vec()
  }
}
