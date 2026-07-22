#![doc = include_str!("../README.md")]

pub mod cartesian;
pub mod composition;

pub use composition::Composition;

/// The degree of a graded structure: the dimension of a simplex, the grade of an
/// exterior form, the degree of a cochain --- one $ZZ$-grading index. The de
/// Rham complex is graded by it; the boundary lowers it by one, the exterior
/// derivative raises it.
///
/// A signed integer, so a value outside $[0, n]$ names a *trivial* space at the
/// end of a finite complex ($Lambda^(-1) = Lambda^(n+1) = 0$). That totality at
/// the degenerate boundary is the point: the codifferential of a $0$-form and
/// the differential of an $n$-form both land in an empty space rather than
/// underflowing. [`Self::index_in`] is the total accessor into a structure of a
/// given top degree, `None` off the range --- exactly the shape of
/// `RoleDim::dim_in`.
///
/// `Dim` and [`ExteriorGrade`](Degree) are aliases: the simplex-dimension and
/// form-grade vocabulary for the one type. Accessors keep the domain word
/// (`dim()`, `grade()`); the type is what unifies them.
///
/// The type follows one pattern, worth naming because it recurs wherever an
/// index space has a degenerate boundary: *totalize the arithmetic, relationize
/// the bound, trivialize the out-of-range*. The representation is a full $ZZ$,
/// so `+`/`-` are total and a computation may pass through $-1$ or $n+1$ with no
/// special case; validity is *not* baked into the representation (as it would be
/// in an unsigned type) but checked *relationally* against a supplied top degree
/// at the point of use ([`Self::index_in`], `None` off range); and a value off
/// $[0, n]$ denotes the trivial object rather than trapping or saturating. This
/// is the pragmatic encoding of what a dependent type would carry as a proof
/// (`Fin (n+1)`): the bound is runtime and non-local -- a degree does not know
/// its own $n$ -- so it cannot live in the type, and the `Option` at the
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
/// Construction is one-directional: an integer lifts *into* a `Degree`, never
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

pub fn binomial(n: usize, k: usize) -> usize {
  num_integer::binomial(n, k)
}
pub fn factorial(num: usize) -> usize {
  (1..=num).product()
}
pub fn factorial_f64(num: usize) -> f64 {
  factorial(num) as f64
}

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
    *self = self.other();
  }

  pub fn as_i32(self) -> i32 {
    self as i32
  }
  pub fn as_f64(self) -> f64 {
    f64::from(self as i32)
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
    self.other()
  }
}
impl std::ops::Mul for Sign {
  type Output = Self;
  fn mul(self, other: Self) -> Self::Output {
    Self::from_bool(self == other)
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

/// Sorts `a` in place and returns the sign (parity) of the sorting
/// permutation.
pub fn sort_signed<T: Ord>(a: &mut [T]) -> Sign {
  Sign::from_parity(sort_count_swaps(a))
}

/// Sorts `a` in place and returns the number of swaps (adjacent
/// transpositions) performed.
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

/// A strictly increasing multi-index: a finite set of indices `0..64`,
/// stored as a bitset.
///
/// The canonical basis element of simplicial chains (a set of vertices) and
/// of exterior algebra blades (a set of covector indices).
///
/// The derived `Ord` compares the bitsets numerically, which for equal
/// cardinality is exactly the colexicographic order.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Combination(u64);

/// The maximum index (exclusive) a [`Combination`] can contain.
pub const MAX_NINDICES: usize = 64;

impl Combination {
  pub fn empty() -> Self {
    Self(0)
  }
  pub fn single(index: usize) -> Self {
    assert!(index < MAX_NINDICES);
    Self(1 << index)
  }
  /// The full set ${0, dots, card - 1}$.
  pub fn full(card: usize) -> Self {
    assert!(card <= MAX_NINDICES);
    if card == MAX_NINDICES {
      Self(u64::MAX)
    } else {
      Self((1 << card) - 1)
    }
  }
  /// From strictly increasing indices.
  pub fn from_increasing(indices: impl IntoIterator<Item = usize>) -> Self {
    let mut set = 0u64;
    for index in indices {
      assert!(index < MAX_NINDICES);
      let bit = 1 << index;
      assert!(
        set & bit == 0 && set < bit,
        "Indices must be strictly increasing."
      );
      set |= bit;
    }
    Self(set)
  }

  /// Canonicalize an arbitrarily ordered index word into the sign of its
  /// permutation and the underlying combination.
  ///
  /// `None` if an index repeats. The only place unsorted multi-indices
  /// exist: as transient inputs.
  pub fn from_word(word: impl IntoIterator<Item = usize>) -> Option<(Sign, Self)> {
    let mut set = 0u64;
    let mut inversions = 0;
    for index in word {
      assert!(index < MAX_NINDICES);
      let bit = 1u64 << index;
      if set & bit != 0 {
        return None;
      }
      // Number of already inserted indices greater than this one.
      inversions += (set >> index).count_ones() as usize;
      set |= bit;
    }
    Some((Sign::from_parity(inversions), Self(set)))
  }

  pub fn bits(self) -> u64 {
    self.0
  }
  pub fn card(self) -> usize {
    self.0.count_ones() as usize
  }
  pub fn is_empty(self) -> bool {
    self.0 == 0
  }
  pub fn contains(self, index: usize) -> bool {
    index < MAX_NINDICES && self.0 & (1 << index) != 0
  }
  pub fn is_subset_of(self, other: Self) -> bool {
    self.0 & other.0 == self.0
  }

  /// The indices in ascending order.
  pub fn iter(self) -> impl Iterator<Item = usize> {
    let mut set = self.0;
    std::iter::from_fn(move || {
      (set != 0).then(|| {
        let index = set.trailing_zeros() as usize;
        set &= set - 1;
        index
      })
    })
  }
  /// The position-th smallest index.
  pub fn index_at(self, position: usize) -> usize {
    self.iter().nth(position).expect("Position out of bounds.")
  }
  /// With the index added; must not be contained yet.
  pub fn inserted(self, index: usize) -> Self {
    assert!(index < MAX_NINDICES && !self.contains(index));
    Self(self.0 | 1 << index)
  }
  /// The position of an index within the set.
  pub fn position_of(self, index: usize) -> usize {
    assert!(self.contains(index));
    (self.0 & ((1 << index) - 1)).count_ones() as usize
  }

  /// Colexicographic rank among all combinations of the same cardinality:
  /// the combinatorial number system $sum_i binom(s_i, i+1)$.
  ///
  /// Independent of any ambient dimension. Ranks of combinations inside
  /// ${0, dots, n-1}$ are exactly $0..binom(n, "card")$.
  pub fn rank(self) -> usize {
    self
      .iter()
      .enumerate()
      .map(|(position, index)| binomial(index, position + 1))
      .sum()
  }
  /// Inverse of [`Self::rank`]: greedy from the largest element.
  pub fn from_rank(card: usize, mut rank: usize) -> Self {
    let mut set = 0u64;
    for position in (1..=card).rev() {
      let mut index = position - 1;
      while binomial(index + 1, position) <= rank {
        index += 1;
      }
      rank -= binomial(index, position);
      set |= 1 << index;
    }
    Self(set)
  }

  /// All combinations of the given cardinality in colexicographic order.
  ///
  /// The universal enumeration: take the first $binom(n, "card")$ to get
  /// exactly the combinations inside ${0, dots, n-1}$.
  pub fn all(card: usize) -> impl Iterator<Item = Self> {
    let mut next = Some(Self::full(card));
    std::iter::from_fn(move || {
      let current = next?;
      next = current.colex_successor();
      Some(current)
    })
  }
  /// The next combination of the same cardinality in colexicographic order
  /// (Gosper's hack).
  fn colex_successor(self) -> Option<Self> {
    let x = self.0;
    if x == 0 {
      return None;
    }
    let u = x & x.wrapping_neg();
    let v = x.checked_add(u)?;
    if v == 0 {
      return None;
    }
    Some(Self(v | (((x ^ v) / u) >> 2)))
  }

  /// Merge two disjoint combinations with the sign of the interleaving
  /// permutation: the wedge of basis blades. `None` if they intersect.
  pub fn union_signed(self, other: Self) -> Option<(Sign, Self)> {
    if self.0 & other.0 != 0 {
      return None;
    }
    let mut inversions = 0;
    for index in other.iter() {
      inversions += (self.0 >> index >> 1).count_ones() as usize;
    }
    Some((Sign::from_parity(inversions), Self(self.0 | other.0)))
  }

  /// The complement within ${0, dots, n-1}$ and the sign such that
  /// $e_S wedge e_(S^c) = "sign" dot e_({0, dots, n-1})$: the combinatorics
  /// of the Hodge star.
  pub fn complement_signed(self, n: usize) -> (Sign, Self) {
    let complement = Self(!self.0 & Self::full(n).0);
    let (sign, _) = self
      .union_signed(complement)
      .expect("Complement is disjoint.");
    (sign, complement)
  }

  /// Alternating single-element deletions $(-1)^i (S without s_i)$:
  /// the boundary of a simplex and the interior product of a blade.
  pub fn deletions(self) -> impl Iterator<Item = (Sign, usize, Self)> {
    self.iter().enumerate().map(move |(position, index)| {
      let deleted = Self(self.0 & !(1 << index));
      (Sign::from_parity(position), index, deleted)
    })
  }

  /// All subsets of the given cardinality, in colexicographic order.
  pub fn subsets(self, card: usize) -> impl Iterator<Item = Self> {
    let ncombinations = binomial(self.card(), card);
    Self::all(card)
      .take(ncombinations)
      .map(move |positions| self.select(positions))
  }
  /// The subset at the given positions: the image of a combination of
  /// positions under the monotone map onto this set's elements.
  pub fn select(self, positions: Self) -> Self {
    Self::from_increasing(positions.iter().map(|position| self.index_at(position)))
  }
}

/// All combinations of cardinality `card` inside ${0, dots, n-1}$,
/// in colexicographic order.
pub fn combinations(n: usize, card: usize) -> impl Iterator<Item = Combination> {
  Combination::all(card).take(binomial(n, card))
}

impl FromIterator<usize> for Combination {
  /// From strictly increasing indices.
  fn from_iter<T: IntoIterator<Item = usize>>(iter: T) -> Self {
    Self::from_increasing(iter)
  }
}

impl std::fmt::Debug for Combination {
  fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
    f.debug_set().entries(self.iter()).finish()
  }
}

#[cfg(test)]
mod test {
  use super::*;

  use itertools::Itertools;

  #[test]
  fn colex_enumeration_and_rank_are_inverse() {
    for card in 0..=5 {
      for (rank, combination) in Combination::all(card).take(100).enumerate() {
        assert_eq!(combination.card(), card);
        assert_eq!(combination.rank(), rank);
        assert_eq!(Combination::from_rank(card, rank), combination);
      }
    }
  }

  /// Colex order is the numeric order of the bitsets and agrees with
  /// comparing the largest elements first.
  #[test]
  fn colex_is_bitset_order() {
    let all: Vec<_> = combinations(6, 3).collect();
    assert!(all.windows(2).all(|w| w[0] < w[1]));
    let mut relexed = all.clone();
    relexed.sort_by_key(|c| {
      let mut descending: Vec<_> = c.iter().collect();
      descending.reverse();
      descending
    });
    assert_eq!(all, relexed);
  }

  /// The first binom(n, k) combinations are exactly those inside 0..n.
  #[test]
  fn colex_enumeration_is_filtration_compatible() {
    for n in 0..=6 {
      for card in 0..=n {
        let inside: Vec<_> = combinations(n, card).collect();
        assert_eq!(inside.len(), binomial(n, card));
        assert!(inside.iter().all(|c| c.iter().all(|index| index < n)));
        assert_eq!(
          inside,
          itertools::Itertools::combinations(0..n, card)
            .map(Combination::from_increasing)
            .sorted()
            .collect::<Vec<_>>()
        );
      }
    }
  }

  #[test]
  fn from_word_canonicalizes() {
    assert_eq!(
      Combination::from_word([2, 0, 1]),
      Some((Sign::Pos, Combination::from_increasing([0, 1, 2])))
    );
    assert_eq!(
      Combination::from_word([1, 0]),
      Some((Sign::Neg, Combination::from_increasing([0, 1])))
    );
    assert_eq!(Combination::from_word([0, 1, 0]), None);
  }

  /// Antisymmetry of the wedge of blades.
  #[test]
  fn union_signed_antisymmetry() {
    let a = Combination::from_increasing([0, 2]);
    let b = Combination::from_increasing([1, 3]);
    let (sign_ab, ab) = a.union_signed(b).unwrap();
    let (sign_ba, ba) = b.union_signed(a).unwrap();
    assert_eq!(ab, ba);
    // grades 2 and 2: sign flip (-1)^(2*2) = +1
    assert_eq!(sign_ab, sign_ba);

    let a = Combination::single(1);
    let b = Combination::single(0);
    let (sign_ab, _) = a.union_signed(b).unwrap();
    let (sign_ba, _) = b.union_signed(a).unwrap();
    assert_eq!(sign_ab, -sign_ba);

    assert_eq!(a.union_signed(a), None);
  }

  /// $e_S wedge e_(S^c) = sign dot e_"full"$ consistency.
  #[test]
  fn complement_signed_wedges_to_top() {
    for n in 0..=6 {
      for card in 0..=n {
        for combination in combinations(n, card) {
          let (sign, complement) = combination.complement_signed(n);
          let (union_sign, union) = combination.union_signed(complement).unwrap();
          assert_eq!(union, Combination::full(n));
          assert_eq!(sign, union_sign);
        }
      }
    }
  }

  /// Double deletions cancel in pairs: $diff compose diff = 0$ at the level
  /// of a single combination.
  #[test]
  fn deletions_square_to_zero() {
    use std::collections::HashMap;
    let combination = Combination::from_increasing([0, 2, 3, 5]);
    let mut chain: HashMap<Combination, i32> = HashMap::new();
    for (sign1, _, face) in combination.deletions() {
      for (sign2, _, subface) in face.deletions() {
        *chain.entry(subface).or_default() += (sign1 * sign2).as_i32();
      }
    }
    assert!(chain.values().all(|&coefficient| coefficient == 0));
  }

  #[test]
  fn select_and_positions() {
    let set = Combination::from_increasing([1, 4, 6]);
    assert_eq!(set.index_at(0), 1);
    assert_eq!(set.index_at(2), 6);
    assert_eq!(set.position_of(4), 1);
    assert_eq!(
      set.select(Combination::from_increasing([0, 2])),
      Combination::from_increasing([1, 6])
    );
    let subsets: Vec<_> = set.subsets(2).collect();
    assert_eq!(
      subsets,
      vec![
        Combination::from_increasing([1, 4]),
        Combination::from_increasing([1, 6]),
        Combination::from_increasing([4, 6]),
      ]
    );
  }
}
