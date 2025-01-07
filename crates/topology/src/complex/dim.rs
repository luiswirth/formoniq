use crate::Dim;

use std::{fmt::Debug, hash::Hash};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum DimInfo {
  Dim(usize),
  Codim(usize),
}
impl DimInfoProvider for DimInfo {
  fn dim_info(self) -> DimInfo {
    self
  }
}

impl DimInfoProvider for Dim {
  fn dim_info(self) -> DimInfo {
    DimInfo::Dim(self)
  }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct ConstDim<const N: usize>;
impl<const N: usize> DimInfoProvider for ConstDim<N> {
  fn dim_info(self) -> DimInfo {
    DimInfo::Dim(N)
  }
}
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct ConstCodim<const N: usize>;
impl<const N: usize> DimInfoProvider for ConstCodim<N> {
  fn dim_info(self) -> DimInfo {
    DimInfo::Codim(N)
  }
}

pub trait DimInfoProvider: Debug + Copy + Eq + Hash {
  fn dim_info(self) -> DimInfo;
  fn dim(self, complex_dim: usize) -> usize {
    match self.dim_info() {
      DimInfo::Dim(d) => d,
      DimInfo::Codim(c) => complex_dim - c,
    }
  }
  fn try_dim(&self, complex_dim: usize) -> Option<usize> {
    self.is_valid(complex_dim).then(|| self.dim(complex_dim))
  }
  fn is_valid(self, complex_dim: usize) -> bool {
    let n = match self.dim_info() {
      DimInfo::Dim(d) => d,
      DimInfo::Codim(c) => c,
    };
    n <= complex_dim
  }
  fn assert_valid(self, complex_dim: usize) {
    assert!(self.is_valid(complex_dim));
  }
}
