//! The $ZZ$-grading index of a graded structure.

/// The degree of a graded structure: the dimension of a simplex, the grade of an
/// exterior form, the degree of a cochain, one $ZZ$-grading index. The de
/// Rham complex is graded by it. The boundary lowers it by one, the exterior
/// derivative raises it.
///
/// A signed integer, so a value outside $[0, n]$ names a trivial space at the
/// end of a finite complex ($Lambda^(-1) = Lambda^(n+1) = 0$). That totality at
/// the degenerate boundary is the point: the codifferential of a $0$-form and
/// the differential of an $n$-form both land in an empty space rather than
/// underflowing. [`Self::index_in`] is the total accessor into a structure of a
/// given top degree, `None` off the range, exactly the shape of
/// `RoleDim::dim_in`.
///
/// [`Dim`] and `ExteriorGrade` are aliases: the simplex-dimension and
/// form-grade vocabulary for the one type. Accessors keep the domain word
/// (`dim()`, `grade()`). The type is what unifies them.
///
/// The type follows one pattern, worth naming because it recurs wherever an
/// index space has a degenerate boundary: totalize the arithmetic, relationize
/// the bound, trivialize the out-of-range. The representation is a full $ZZ$,
/// so `+`/`-` are total and a computation may pass through $-1$ or $n+1$ with no
/// special case; validity is not baked into the representation (as it would be
/// in an unsigned type) but checked relationally against a supplied top degree
/// at the point of use ([`Self::index_in`], `None` off range); and a value off
/// $[0, n]$ denotes the trivial object rather than trapping or saturating. This
/// is the pragmatic encoding of what a dependent type would carry as a proof
/// (`Fin (n+1)`): the bound is runtime and non-local: a degree does not know
/// its own $n$, so it cannot live in the type, and the `Option` at the
/// boundary is where it lives instead.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Degree(i64);

impl Degree {
  pub const ZERO: Degree = Degree(0);
  pub const ONE: Degree = Degree(1);

  pub const fn new(k: i64) -> Self {
    Self(k)
  }
  /// The raw signed index.
  pub const fn get(self) -> i64 {
    self.0
  }
  /// The index as a `usize`, for a degree known non-negative. Panics on a
  /// negative degree; use [`Self::index_in`] where the trivial ends are
  /// reachable.
  pub fn index(self) -> usize {
    usize::try_from(self.0).expect("negative degree has no usize index")
  }
  /// The `usize` index into a graded structure of top degree `top`, `None`
  /// outside $[0, "top"]$ where the space is trivial.
  pub fn index_in(self, top: Degree) -> Option<usize> {
    (self.0 >= 0 && self.0 <= top.0).then_some(self.0 as usize)
  }
  /// Whether the degree names a non-trivial space of a complex of top degree
  /// `top`, i.e. lies in $[0, "top"]$.
  pub fn in_range(self, top: Degree) -> bool {
    self.0 >= 0 && self.0 <= top.0
  }
  pub fn is_zero(self) -> bool {
    self.0 == 0
  }
  /// The degrees $0, 1, dots, "self"$ ascending.
  pub fn range_inclusive(self) -> impl DoubleEndedIterator<Item = Degree> + Clone {
    (0..=self.0).map(Degree)
  }
  /// The degrees $"self", dots, "other"$ ascending; empty if `other` is below.
  pub fn range_to_inclusive(
    self,
    other: Degree,
  ) -> impl DoubleEndedIterator<Item = Degree> + Clone {
    (self.0..=other.0).map(Degree)
  }
  /// The degrees $0, 1, dots, "self" - 1$ ascending.
  pub fn range(self) -> impl DoubleEndedIterator<Item = Degree> + Clone {
    (0..self.0).map(Degree)
  }
}

/// A [`Degree`] is constructed freely from any integer: `usize` for the counts
/// that name it in practice, signed types so a bare literal (which defaults to
/// `i32`) lifts with no annotation and `(-1).into()` names the trivial degree.
/// Construction is one-directional: an integer lifts into a `Degree`, never
/// the reverse, so the signed grading logic stays sealed inside the type.
macro_rules! impl_degree_from_int {
  ($($t:ty),*) => {$(
    impl From<$t> for Degree {
      fn from(k: $t) -> Self {
        Self(k as i64)
      }
    }
  )*};
}
impl_degree_from_int!(usize, u32, u64, isize, i32, i64);

impl std::str::FromStr for Degree {
  type Err = std::num::ParseIntError;
  fn from_str(s: &str) -> Result<Self, Self::Err> {
    s.parse::<i64>().map(Degree)
  }
}

impl std::ops::Add<usize> for Degree {
  type Output = Degree;
  fn add(self, rhs: usize) -> Degree {
    Degree(self.0 + rhs as i64)
  }
}
impl std::ops::Sub<usize> for Degree {
  type Output = Degree;
  fn sub(self, rhs: usize) -> Degree {
    Degree(self.0 - rhs as i64)
  }
}
impl std::ops::Add for Degree {
  type Output = Degree;
  fn add(self, rhs: Degree) -> Degree {
    Degree(self.0 + rhs.0)
  }
}
impl std::ops::Sub for Degree {
  type Output = Degree;
  fn sub(self, rhs: Degree) -> Degree {
    Degree(self.0 - rhs.0)
  }
}
// Comparisons against a raw count, in both directions: a degree is routinely
// tested against a cardinality or a matrix dimension. Integer literals infer
// `usize` here, so `grade == 0` and `grade + 1` keep reading naturally.
impl PartialEq<usize> for Degree {
  fn eq(&self, rhs: &usize) -> bool {
    self.0 == *rhs as i64
  }
}
impl PartialOrd<usize> for Degree {
  fn partial_cmp(&self, rhs: &usize) -> Option<std::cmp::Ordering> {
    self.0.partial_cmp(&(*rhs as i64))
  }
}
impl PartialEq<Degree> for usize {
  fn eq(&self, rhs: &Degree) -> bool {
    *self as i64 == rhs.0
  }
}
impl PartialOrd<Degree> for usize {
  fn partial_cmp(&self, rhs: &Degree) -> Option<std::cmp::Ordering> {
    (*self as i64).partial_cmp(&rhs.0)
  }
}
impl std::fmt::Debug for Degree {
  fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
    write!(f, "{}", self.0)
  }
}
impl std::fmt::Display for Degree {
  fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
    write!(f, "{}", self.0)
  }
}

/// The dimension of a simplex or space: the [`Degree`] under its geometric name.
pub type Dim = Degree;
