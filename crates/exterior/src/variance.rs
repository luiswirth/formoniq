use std::{fmt::Debug, hash::Hash};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Variance {
  Contra,
  Co,
}

pub trait VarianceTrait: Debug + Copy + Clone + PartialEq + Eq + Hash {
  fn variance(&self) -> Variance;
}
impl VarianceTrait for Variance {
  fn variance(&self) -> Variance {
    *self
  }
}

pub trait VarianceMarker: Debug + Copy + Clone + PartialEq + Eq + Hash {
  fn variance() -> Variance;
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Contra;
impl VarianceMarker for Contra {
  fn variance() -> Variance {
    Variance::Contra
  }
}
impl VarianceTrait for Contra {
  fn variance(&self) -> Variance {
    Variance::Contra
  }
}
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Co;
impl VarianceMarker for Co {
  fn variance() -> Variance {
    Variance::Co
  }
}
impl VarianceTrait for Co {
  fn variance(&self) -> Variance {
    Variance::Co
  }
}
