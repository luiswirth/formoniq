use super::MultiIndex;
use crate::{
  sign::{sort_signed, Sign},
  SignedIndexSet,
};

use std::{fmt::Debug, hash::Hash, marker::PhantomData};

pub trait IndexKind: Debug + Default + Clone + Copy + PartialEq + Eq + Hash {}

/// Arbitrary order of elements. May contain duplicates.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ArbitraryList;
impl IndexKind for ArbitraryList {}

/// Strictly increasing elements! No duplicates allowed.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IncreasingSet;
impl IndexKind for IncreasingSet {}

/// Variant Conversions
impl<O: IndexKind> MultiIndex<O> {
  pub fn assume_sorted(self) -> MultiIndex<IncreasingSet> {
    debug_assert!(self.indices.is_sorted_by(|a, b| a < b));
    MultiIndex {
      indices: self.indices,
      _order: PhantomData,
    }
  }

  pub fn is_nodup(&self) -> bool {
    self.permut_sign().is_some()
  }
  pub fn permut_sign(&self) -> Option<Sign> {
    self.clone().try_into_sorted_signed().map(|s| s.sign)
  }

  pub fn try_into_sorted_signed(self) -> Option<SignedIndexSet<IncreasingSet>> {
    self.with_sign(Sign::Pos).try_into_sorted_signed()
  }
  pub fn into_sorted_signed(self) -> SignedIndexSet<IncreasingSet> {
    self.try_into_sorted_signed().unwrap()
  }
  pub fn into_sorted(self) -> MultiIndex<IncreasingSet> {
    self.into_sorted_signed().set
  }
}

impl<O: IndexKind> SignedIndexSet<O> {
  pub fn into_sorted_signed(self) -> SignedIndexSet<IncreasingSet> {
    self.try_into_sorted_signed().unwrap()
  }

  /// Returns [`None`] if there is a duplicate index.
  pub fn try_into_sorted_signed(self) -> Option<SignedIndexSet<IncreasingSet>> {
    let mut indices = self.set.indices;

    let sort_sign = sort_signed(&mut indices);
    let sign = self.sign * sort_sign;

    let len = indices.len();
    indices.dedup();
    if indices.len() != len {
      return None;
    }

    Some(
      MultiIndex {
        indices,
        _order: PhantomData,
      }
      .with_sign(sign),
    )
  }
}

impl<O: IndexKind> MultiIndex<O> {
  pub fn with_sign(self, sign: Sign) -> SignedIndexSet<O> {
    SignedIndexSet::new(self, sign)
  }
}
