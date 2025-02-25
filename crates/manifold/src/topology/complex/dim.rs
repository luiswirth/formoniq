use crate::Dim;

use std::{fmt::Debug, hash::Hash};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum RelDim {
  Dim(usize),
  Codim(usize),
}
impl RelDim {
  pub fn abs_dim(&self, total_dim: Dim) -> Dim {
    match self {
      RelDim::Dim(d) => *d,
      RelDim::Codim(c) => total_dim - c,
    }
  }
}
impl Default for RelDim {
  fn default() -> Self {
    Self::Dim(0)
  }
}
