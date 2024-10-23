extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

pub mod assemble;
pub mod combinatorics;
pub mod fe;
pub mod geometry;
pub mod matrix;
pub mod mesh;
pub mod space;
pub mod util;

pub type Dim = usize;
pub type Codim = usize;

pub type Length = f64;

pub type VertexIdx = usize;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Orientation {
  #[default]
  Pos = 1,
  Neg = -1,
}

impl Orientation {
  /// Simplex orientation might change when permuting the vertices.
  /// This depends on the parity of the number of swaps.
  /// Even permutations preserve the orientation.
  /// Odd permutations invert the orientation.
  /// Based on the number
  pub fn from_permutation_parity(n: usize) -> Self {
    match n % 2 {
      0 => Self::Pos,
      1 => Self::Neg,
      _ => unreachable!(),
    }
  }

  pub fn other(self) -> Self {
    match self {
      Orientation::Pos => Orientation::Neg,
      Orientation::Neg => Orientation::Pos,
    }
  }
  pub fn switch(&mut self) {
    *self = self.other()
  }

  pub fn as_i32(self) -> i32 {
    self as i32
  }
  pub fn as_f64(self) -> f64 {
    self as i32 as f64
  }
}
impl std::ops::Neg for Orientation {
  type Output = Self;

  fn neg(self) -> Self::Output {
    match self {
      Self::Pos => Self::Neg,
      Self::Neg => Self::Pos,
    }
  }
}
impl std::ops::Mul for Orientation {
  type Output = Self;

  fn mul(self, other: Self) -> Self::Output {
    match self == other {
      true => Self::Pos,
      false => Self::Neg,
    }
  }
}
impl std::ops::MulAssign for Orientation {
  fn mul_assign(&mut self, other: Self) {
    *self = *self * other;
  }
}
impl From<Orientation> for char {
  fn from(o: Orientation) -> Self {
    match o {
      Orientation::Pos => '+',
      Orientation::Neg => '-',
    }
  }
}
impl std::fmt::Display for Orientation {
  fn fmt(&self, fmt: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
    write!(fmt, "{}", char::from(*self))
  }
}
