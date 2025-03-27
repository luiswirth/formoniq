#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum Sign {
  #[default]
  Pos = 1,
  Neg = -1,
}

impl Sign {
  pub fn from_bool(b: bool) -> Self {
    match b {
      true => Self::Pos,
      false => Self::Neg,
    }
  }
  pub fn from_f64(f: f64) -> Option<Self> {
    if f == 0.0 {
      return None;
    }
    Some(Self::from_bool(f > 0.0))
  }

  /// permutation parity
  pub fn from_parity(n: usize) -> Self {
    match n % 2 {
      0 => Self::Pos,
      1 => Self::Neg,
      _ => unreachable!(),
    }
  }

  pub fn other(self) -> Self {
    match self {
      Sign::Pos => Sign::Neg,
      Sign::Neg => Sign::Pos,
    }
  }
  pub fn flip(&mut self) {
    *self = self.other()
  }

  pub fn as_i32(self) -> i32 {
    self as i32
  }
  pub fn as_f64(self) -> f64 {
    self as i32 as f64
  }

  pub fn is_pos(self) -> bool {
    self == Self::Pos
  }
  pub fn is_neg(self) -> bool {
    self == Self::Neg
  }
}
impl std::ops::Neg for Sign {
  type Output = Self;
  fn neg(self) -> Self::Output {
    match self {
      Self::Pos => Self::Neg,
      Self::Neg => Self::Pos,
    }
  }
}
impl std::ops::Mul for Sign {
  type Output = Self;
  fn mul(self, other: Self) -> Self::Output {
    match self == other {
      true => Self::Pos,
      false => Self::Neg,
    }
  }
}
impl std::ops::MulAssign for Sign {
  fn mul_assign(&mut self, other: Self) {
    *self = *self * other;
  }
}
impl From<Sign> for char {
  fn from(o: Sign) -> Self {
    match o {
      Sign::Pos => '+',
      Sign::Neg => '-',
    }
  }
}
impl std::fmt::Display for Sign {
  fn fmt(&self, fmt: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
    write!(fmt, "{}", char::from(*self))
  }
}

/// Returns the sorted permutation of `a` and the sign of the permutation.
pub fn sort_signed<T: Ord>(a: &mut [T]) -> Sign {
  Sign::from_parity(sort_count_swaps(a))
}

/// Returns the sorted permutation of `a` and the number of swaps.
pub fn sort_count_swaps<T: Ord>(a: &mut [T]) -> usize {
  let mut nswaps = 0;

  let mut n = a.len();
  if n > 0 {
    let mut swapped = true;
    while swapped {
      swapped = false;
      for i in 1..n {
        if a[i - 1] > a[i] {
          a.swap(i - 1, i);
          swapped = true;
          nswaps += 1;
        }
      }
      n -= 1;
    }
  }
  nswaps
}
