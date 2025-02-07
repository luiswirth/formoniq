use crate::Dim;

use std::{fmt::Debug, hash::Hash};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum RelDim {
  Dim(usize),
  Codim(usize),
}
impl Default for RelDim {
  fn default() -> Self {
    Self::Dim(0)
  }
}
impl RelDimTrait for RelDim {
  fn rel_dim(self) -> RelDim {
    self
  }
}

impl RelDimTrait for Dim {
  fn rel_dim(self) -> RelDim {
    RelDim::Dim(self)
  }
}

#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub struct ConstDim<const N: usize>;
impl<const N: usize> RelDimTrait for ConstDim<N> {
  fn rel_dim(self) -> RelDim {
    RelDim::Dim(N)
  }
}
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub struct ConstCodim<const N: usize>;
impl<const N: usize> RelDimTrait for ConstCodim<N> {
  fn rel_dim(self) -> RelDim {
    RelDim::Codim(N)
  }
}

pub trait RelDimTrait: Debug + Default + Copy + Eq + Hash {
  fn rel_dim(self) -> RelDim;
  fn dim(self, complex_dim: usize) -> usize {
    match self.rel_dim() {
      RelDim::Dim(d) => d,
      RelDim::Codim(c) => complex_dim - c,
    }
  }
  fn try_dim(&self, complex_dim: usize) -> Option<usize> {
    self.is_valid(complex_dim).then(|| self.dim(complex_dim))
  }
  fn is_valid(self, complex_dim: usize) -> bool {
    let n = match self.rel_dim() {
      RelDim::Dim(d) => d,
      RelDim::Codim(c) => c,
    };
    n <= complex_dim
  }
  fn assert_valid(self, complex_dim: usize) {
    assert!(self.is_valid(complex_dim));
  }
}
