use super::IndexSet;
use crate::combo::{sort_signed, Sign};

pub trait Base: Clone {}
pub trait Specified: Base {
  fn n(&self) -> usize;
  fn indices(&self) -> Vec<usize>;
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Unspecified;
impl Base for Unspecified {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Local(pub usize);
impl Base for Local {}
impl Specified for Local {
  fn n(&self) -> usize {
    self.0
  }
  fn indices(&self) -> Vec<usize> {
    (0..self.0).collect()
  }
}
impl From<usize> for Local {
  fn from(value: usize) -> Self {
    Self(value)
  }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Global(pub Vec<usize>);
impl Base for Global {}
impl Specified for Global {
  fn n(&self) -> usize {
    self.0.len()
  }
  fn indices(&self) -> Vec<usize> {
    self.0.clone()
  }
}
impl From<Vec<usize>> for Global {
  fn from(value: Vec<usize>) -> Self {
    Self(value)
  }
}

pub trait Order: Clone + Copy {}
impl Order for Sorted {}
impl Order for Ordered {}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Sorted;
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Ordered;

pub trait Signedness: Clone + Copy {
  fn get_or_default(&self) -> Sign;
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Unsigned;
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Signed(pub Sign);
impl Signedness for Unsigned {
  fn get_or_default(&self) -> Sign {
    Sign::default()
  }
}
impl Signedness for Signed {
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
impl<B: Base, O: Order, S: Signedness> IndexSet<B, O, S> {
  pub fn with_local_base(self, local: impl Into<Local>) -> IndexSet<Local, O, S> {
    let local = local.into();
    assert!(self.iter().all(|i| *i < local.0));
    IndexSet {
      indices: self.indices,
      base: local,
      order: self.order,
      signedness: self.signedness,
    }
  }
  pub fn with_global_base(self, global: impl Into<Global>) -> IndexSet<Global, O, S> {
    IndexSet {
      indices: self.indices,
      base: global.into(),
      order: self.order,
      signedness: self.signedness,
    }
  }

  pub fn forget_base(self) -> IndexSet<Unspecified, O, S> {
    IndexSet {
      indices: self.indices,
      base: Unspecified,
      order: self.order,
      signedness: self.signedness,
    }
  }

  pub fn assume_sorted(self) -> IndexSet<B, Sorted, S> {
    debug_assert!(self.indices.is_sorted());
    IndexSet {
      indices: self.indices,
      base: self.base,
      order: Sorted,
      signedness: self.signedness,
    }
  }
  pub fn sort_sign(self) -> IndexSet<B, Sorted, Signed> {
    let mut indices = self.indices;
    let sort_sign = sort_signed(&mut indices);
    let self_sign = self.signedness.get_or_default();
    let sign = self_sign * sort_sign;
    IndexSet {
      indices,
      base: self.base,
      order: Sorted,
      signedness: Signed(sign),
    }
  }
  pub fn forget_sorted(self) -> IndexSet<B, Ordered, S> {
    IndexSet {
      indices: self.indices,
      base: self.base,
      order: Ordered,
      signedness: self.signedness,
    }
  }

  pub fn with_sign(self, sign: impl Into<Signed>) -> IndexSet<B, O, Signed> {
    IndexSet {
      indices: self.indices,
      base: self.base,
      order: self.order,
      signedness: sign.into(),
    }
  }
  pub fn forget_sign(self) -> IndexSet<B, O, Unsigned> {
    IndexSet {
      indices: self.indices,
      base: self.base,
      order: self.order,
      signedness: Unsigned,
    }
  }
}
