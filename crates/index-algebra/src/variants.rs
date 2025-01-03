use super::IndexSet;
use crate::sign::{sort_signed, Sign};

use std::fmt::Debug;

pub trait SetOrder: Clone + Copy + Eq {}
impl SetOrder for ArbitraryOrder {}
impl SetOrder for CanonicalOrder {}

/// Arbitrary order of elements. May contain duplicates.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ArbitraryOrder;

/// Strictly increasing elements! No duplicates allowed.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CanonicalOrder;

pub trait SetSign: Clone + Copy {
  fn get_or_default(&self) -> Sign;
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Unsigned;
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Signed(pub Sign);
impl SetSign for Unsigned {
  fn get_or_default(&self) -> Sign {
    Sign::default()
  }
}
impl SetSign for Signed {
  fn get_or_default(&self) -> Sign {
    self.0
  }
}
impl From<Sign> for Signed {
  fn from(value: Sign) -> Self {
    Signed(value)
  }
}

/// Variant Conversions
impl<O: SetOrder, S: SetSign> IndexSet<O, S> {
  pub fn assume_sorted(self) -> IndexSet<CanonicalOrder, S> {
    debug_assert!(self.indices.is_sorted_by(|a, b| a < b));
    IndexSet {
      indices: self.indices,
      order: CanonicalOrder,
      signedness: self.signedness,
    }
  }

  pub fn sort(self) -> IndexSet<CanonicalOrder, Unsigned> {
    self.sort_signed().forget_sign()
  }

  pub fn sort_signed(self) -> IndexSet<CanonicalOrder, Signed> {
    self.try_sort_signed().unwrap()
  }

  /// Returns [`None`] if there is a duplicate index.
  pub fn try_sort_signed(self) -> Option<IndexSet<CanonicalOrder, Signed>> {
    let mut indices = self.indices;
    let sort_sign = sort_signed(&mut indices);
    let self_sign = self.signedness.get_or_default();
    let sign = self_sign * sort_sign;

    let len = indices.len();
    indices.dedup();
    if indices.len() != len {
      return None;
    }

    Some(IndexSet {
      indices,
      order: CanonicalOrder,
      signedness: Signed(sign),
    })
  }

  pub fn into_ordered(self) -> IndexSet<ArbitraryOrder, S> {
    IndexSet {
      indices: self.indices,
      order: ArbitraryOrder,
      signedness: self.signedness,
    }
  }

  pub fn with_sign(self, sign: impl Into<Sign>) -> IndexSet<O, Signed> {
    IndexSet {
      indices: self.indices,
      order: self.order,
      signedness: Signed(sign.into()),
    }
  }
  pub fn forget_sign(self) -> IndexSet<O, Unsigned> {
    IndexSet {
      indices: self.indices,
      order: self.order,
      signedness: Unsigned,
    }
  }

  pub fn into_oriented(self) -> IndexSet<ArbitraryOrder, Signed> {
    IndexSet {
      indices: self.indices,
      order: ArbitraryOrder,
      signedness: Signed(Sign::Pos),
    }
  }
}
