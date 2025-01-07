use super::IndexSet;
use crate::{
  sign::{sort_signed, Sign},
  SignedIndexSet,
};

use std::fmt::Debug;

pub trait SetOrder: Default + Clone + Copy + Eq {}
impl SetOrder for ArbitraryOrder {}
impl SetOrder for CanonicalOrder {}

/// Arbitrary order of elements. May contain duplicates.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ArbitraryOrder;

/// Strictly increasing elements! No duplicates allowed.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CanonicalOrder;

/// Variant Conversions
impl<O: SetOrder> IndexSet<O> {
  pub fn assume_sorted(self) -> IndexSet<CanonicalOrder> {
    debug_assert!(self.indices.is_sorted_by(|a, b| a < b));
    IndexSet {
      indices: self.indices,
      order: CanonicalOrder,
    }
  }

  pub fn is_nodup(&self) -> bool {
    self.permut_sign().is_some()
  }
  pub fn permut_sign(&self) -> Option<Sign> {
    self.clone().try_into_sorted_signed().map(|s| s.sign)
  }

  pub fn try_into_sorted_signed(self) -> Option<SignedIndexSet<CanonicalOrder>> {
    self.with_sign(Sign::Pos).try_into_sorted_signed()
  }
  pub fn into_sorted_signed(self) -> SignedIndexSet<CanonicalOrder> {
    self.try_into_sorted_signed().unwrap()
  }
  pub fn into_sorted(self) -> IndexSet<CanonicalOrder> {
    self.into_sorted_signed().set
  }
}

impl<O: SetOrder> SignedIndexSet<O> {
  pub fn into_sorted_signed(self) -> SignedIndexSet<CanonicalOrder> {
    self.try_into_sorted_signed().unwrap()
  }

  /// Returns [`None`] if there is a duplicate index.
  pub fn try_into_sorted_signed(self) -> Option<SignedIndexSet<CanonicalOrder>> {
    let mut indices = self.set.indices;

    let sort_sign = sort_signed(&mut indices);
    let sign = self.sign * sort_sign;

    let len = indices.len();
    indices.dedup();
    if indices.len() != len {
      return None;
    }

    Some(
      IndexSet {
        indices,
        order: CanonicalOrder,
      }
      .with_sign(sign),
    )
  }
}

impl<O: SetOrder> IndexSet<O> {
  pub fn with_sign(self, sign: Sign) -> SignedIndexSet<O> {
    SignedIndexSet::new(self, sign)
  }
}
