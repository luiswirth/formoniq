#![doc = include_str!("../README.md")]

pub mod cartesian;
pub mod composition;
pub mod monotone;
pub mod permutation;

pub use cartesian::{Word, WordDeletions, Words};
pub use composition::Composition;
pub use monotone::{MonoDeletions, MonoIndex, MonoIndices, Repetition, Symbols};

/// A basis element of one of the three index families.
///
/// The three are the bases of the three symmetry types:
/// [`Combination`] a subset for $Lambda^k$, [`Composition`] an exponent vector
/// for $"Sym"^k$, and [`Word`] a word for $V^(times.circle k)$.
///
/// [`MonoIndex`] covers the two monotone ones, subsets and multisets, which the
/// shift makes a single bitset. [`Word`] covers the free one, where there
/// is no symmetry to quotient by and therefore nothing to compress: the two
/// representations differ because the objects do.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MultiIndex {
  /// A subset or a multiset: the basis of $Lambda^k$ or $"Sym"^k$.
  Mono(MonoIndex),
  /// A word: the basis of the free power $V^(times.circle k)$.
  Word(Word),
}

impl MultiIndex {
  pub fn degree(&self) -> usize {
    match self {
      Self::Mono(index) => index.degree(),
      Self::Word(index) => index.degree(),
    }
  }
  /// The rank in the family's own order: colex for the monotone ones, radix
  /// for the free one.
  pub fn rank(&self) -> usize {
    match self {
      Self::Mono(index) => index.rank(),
      Self::Word(index) => index.rank(),
    }
  }
  /// The symbols, in position order.
  pub fn word(&self) -> Symbols {
    match self {
      Self::Mono(index) => index.word(),
      Self::Word(index) => index.symbols(),
    }
  }
  pub fn symbol(&self, position: usize) -> usize {
    match self {
      Self::Mono(index) => index.symbol(position),
      Self::Word(index) => index.symbol(position),
    }
  }
  /// How many positions carry a symbol.
  pub fn multiplicity(&self, symbol: usize) -> usize {
    match self {
      Self::Mono(index) => index.multiplicity(symbol),
      Self::Word(index) => index.iter().filter(|&s| s == symbol).count(),
    }
  }

  /// The monotone index, if this is one.
  pub fn as_mono(&self) -> Option<&MonoIndex> {
    match self {
      Self::Mono(index) => Some(index),
      Self::Word(_) => None,
    }
  }
  /// The cartesian index, if this is one.
  pub fn as_word(&self) -> Option<&Word> {
    match self {
      Self::Word(index) => Some(index),
      Self::Mono(_) => None,
    }
  }

  /// Combine two indices of one family: the signed merge on the quotients,
  /// concatenation on the free power.
  ///
  /// `None` only where the result is the zero of the algebra, which happens on
  /// an alternating index alone. The symmetric merge and the free
  /// concatenation are both total.
  ///
  /// # Panics
  /// If the families differ.
  pub fn merge(&self, other: &Self) -> Option<(Sign, Self)> {
    match (self, other) {
      (Self::Mono(left), Self::Mono(right)) => {
        left.merge(right).map(|(sign, index)| (sign, index.into()))
      }
      (Self::Word(left), Self::Word(right)) => Some((Sign::Pos, left.concat(right).into())),
      _ => panic!("a merge is within one family"),
    }
  }

  /// Every single-position deletion, with the sign it carries.
  ///
  /// Alternating deletions alternate in sign, symmetric and free ones do not.
  /// A free deletion is the simplest of the three: positions are distinct and
  /// nothing is reordered, so there is no permutation to take a sign from.
  pub fn deletions(&self) -> MultiDeletions {
    match self {
      Self::Mono(index) => MultiDeletions::Mono(index.deletions()),
      Self::Word(index) => MultiDeletions::Word(index.deletions()),
    }
  }
}

/// Every index of a family, degree and alphabet.
///
/// An enum rather than a boxed trait object: the monotone path allocates
/// nothing, and both variants are now the same size, so nothing is paid for
/// the family that is not in use.
#[derive(Debug, Clone)]
pub enum MultiIndices {
  Mono(MonoIndices),
  Word(Words),
}

impl Iterator for MultiIndices {
  type Item = MultiIndex;
  fn next(&mut self) -> Option<MultiIndex> {
    match self {
      Self::Mono(indices) => indices.next().map(MultiIndex::Mono),
      Self::Word(indices) => indices.next().map(MultiIndex::Word),
    }
  }
  fn size_hint(&self) -> (usize, Option<usize>) {
    match self {
      Self::Mono(indices) => indices.size_hint(),
      Self::Word(indices) => indices.size_hint(),
    }
  }
}
impl ExactSizeIterator for MultiIndices {}

/// Every single-position deletion of a [`MultiIndex`], with its sign.
#[derive(Debug, Clone)]
pub enum MultiDeletions {
  Mono(MonoDeletions),
  Word(WordDeletions),
}

impl Iterator for MultiDeletions {
  type Item = (Sign, usize, MultiIndex);
  fn next(&mut self) -> Option<Self::Item> {
    match self {
      Self::Mono(deletions) => deletions
        .next()
        .map(|(sign, symbol, reduced)| (sign, symbol, MultiIndex::Mono(reduced))),
      Self::Word(deletions) => deletions
        .next()
        .map(|(symbol, reduced)| (Sign::Pos, symbol, MultiIndex::Word(reduced))),
    }
  }
}

impl Default for MultiIndex {
  /// The empty alternating index: the basis of the scalars.
  fn default() -> Self {
    Self::Mono(MonoIndex::default())
  }
}

impl From<MonoIndex> for MultiIndex {
  fn from(index: MonoIndex) -> Self {
    Self::Mono(index)
  }
}
impl From<Word> for MultiIndex {
  fn from(index: Word) -> Self {
    Self::Word(index)
  }
}

pub use permutation::Permutation;

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
/// `Dim` and [`ExteriorGrade`](Degree) are aliases: the simplex-dimension and
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

/// Pascal's triangle up to the index ceiling, computed once.
///
/// A rank is a sum of binomials and ranking is the innermost thing the algebra
/// does, so this is a table lookup rather than a division loop. It covers every
/// shifted symbol a [`MonoIndex`] can hold; beyond that the exact computation
/// still runs, so nothing here is a bound on what may be counted.
const BINOMIAL_TABLE_SIZE: usize = monotone::MAX_SHIFTED_SYMBOLS + 1;
static BINOMIALS: std::sync::LazyLock<[[usize; BINOMIAL_TABLE_SIZE]; BINOMIAL_TABLE_SIZE]> =
  std::sync::LazyLock::new(|| {
    let mut table = [[0usize; BINOMIAL_TABLE_SIZE]; BINOMIAL_TABLE_SIZE];
    for n in 0..BINOMIAL_TABLE_SIZE {
      table[n][0] = 1;
      for k in 1..=n {
        table[n][k] = table[n - 1][k - 1].saturating_add(table[n - 1].get(k).copied().unwrap_or(0));
      }
    }
    table
  });

pub fn binomial(n: usize, k: usize) -> usize {
  if n < BINOMIAL_TABLE_SIZE && k < BINOMIAL_TABLE_SIZE {
    BINOMIALS[n][k]
  } else {
    num_integer::binomial(n, k)
  }
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

/// A strictly increasing multi-index: a finite set of indices, stored as the
/// bitset of its symbols.
///
/// The basis element of $Lambda^k$ (a set of covector indices) and of
/// simplicial chains (a set of vertices).
///
/// A newtype over [`MonoIndex`] at [`Repetition::Forbidden`], which is what it
/// has always been: a subset is a monotone word that may not repeat, and
/// forbidding repetition makes the shift zero, so the shifted word a
/// `MonoIndex` stores is the set. The wrapper adds nothing to the
/// representation and enforces the family, so a multiset cannot be handed to a
/// subset's operations.
///
/// The derived `Ord` compares the bitsets numerically, which for equal
/// cardinality is exactly the colexicographic order.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Combination(MonoIndex);

/// The maximum index (exclusive) a [`Combination`] can contain.
pub const MAX_NINDICES: usize = monotone::MAX_SHIFTED_SYMBOLS;

impl Combination {
  pub fn empty() -> Self {
    Self(MonoIndex::empty(Repetition::Forbidden))
  }
  pub fn single(index: usize) -> Self {
    Self(MonoIndex::single(Repetition::Forbidden, index))
  }
  /// The full set ${0, dots, card - 1}$.
  pub fn full(card: usize) -> Self {
    assert!(card <= MAX_NINDICES);
    if card == MAX_NINDICES {
      Self::from_bits(u128::MAX)
    } else {
      Self::from_bits((1u128 << card) - 1)
    }
  }
  /// From strictly increasing indices.
  pub fn from_increasing(indices: impl IntoIterator<Item = usize>) -> Self {
    let mut set = 0u128;
    for index in indices {
      assert!(index < MAX_NINDICES);
      let bit = 1u128 << index;
      assert!(
        set & bit == 0 && set < bit,
        "Indices must be strictly increasing."
      );
      set |= bit;
    }
    Self::from_bits(set)
  }

  /// Canonicalize an arbitrarily ordered index word into the sign of its
  /// permutation and the underlying combination.
  ///
  /// `None` if an index repeats. The only place unsorted multi-indices
  /// exist: as transient inputs.
  pub fn from_word(word: impl IntoIterator<Item = usize>) -> Option<(Sign, Self)> {
    let mut set = 0u128;
    let mut inversions = 0;
    for index in word {
      assert!(index < MAX_NINDICES);
      let bit = 1u128 << index;
      if set & bit != 0 {
        return None;
      }
      // Number of already inserted indices greater than this one.
      inversions += (set >> index).count_ones() as usize;
      set |= bit;
    }
    Some((Sign::from_parity(inversions), Self::from_bits(set)))
  }

  /// The raw bitset, each set bit an index.
  pub fn bits(self) -> u128 {
    self.0.shifted_bits()
  }
  /// From a raw bitset, each set bit an index.
  ///
  /// Every bitset denotes a combination, so this is total. The name marks that
  /// the caller is working at the bit level, where the set is the bits.
  pub fn from_bits(bits: u128) -> Self {
    Self(MonoIndex::from_shifted(Repetition::Forbidden, bits))
  }
  pub fn card(self) -> usize {
    self.0.degree()
  }
  pub fn is_empty(self) -> bool {
    self.card() == 0
  }
  pub fn contains(self, index: usize) -> bool {
    index < MAX_NINDICES && self.bits() & (1 << index) != 0
  }
  pub fn is_subset_of(self, other: Self) -> bool {
    self.bits() & other.bits() == self.bits()
  }

  /// The indices in ascending order.
  pub fn iter(self) -> impl Iterator<Item = usize> {
    monotone::set_bits(self.bits())
  }
  /// The position-th smallest index.
  pub fn index_at(self, position: usize) -> usize {
    self.iter().nth(position).expect("Position out of bounds.")
  }
  /// With the index added; must not be contained yet.
  pub fn inserted(self, index: usize) -> Self {
    assert!(index < MAX_NINDICES && !self.contains(index));
    Self::from_bits(self.bits() | 1 << index)
  }
  /// The position of an index within the set.
  pub fn position_of(self, index: usize) -> usize {
    assert!(self.contains(index));
    (self.bits() & ((1 << index) - 1)).count_ones() as usize
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
    let mut set = 0u128;
    for position in (1..=card).rev() {
      let mut index = position - 1;
      while binomial(index + 1, position) <= rank {
        index += 1;
      }
      rank -= binomial(index, position);
      set |= 1u128 << index;
    }
    Self::from_bits(set)
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
    let x = self.bits();
    if x == 0 {
      return None;
    }
    let u = x & x.wrapping_neg();
    let v = x.checked_add(u)?;
    if v == 0 {
      return None;
    }
    Some(Self::from_bits(v | (((x ^ v) / u) >> 2)))
  }

  /// Merge two disjoint combinations with the sign of the interleaving
  /// permutation: the wedge of basis blades. `None` if they intersect.
  pub fn union_signed(self, other: Self) -> Option<(Sign, Self)> {
    if self.bits() & other.bits() != 0 {
      return None;
    }
    let mut inversions = 0;
    for index in other.iter() {
      inversions += (self.bits() >> index >> 1).count_ones() as usize;
    }
    Some((
      Sign::from_parity(inversions),
      Self::from_bits(self.bits() | other.bits()),
    ))
  }

  /// The complement within ${0, dots, n-1}$ and the sign such that
  /// $e_S wedge e_(S^c) = "sign" dot e_({0, dots, n-1})$: the combinatorics
  /// of the Hodge star.
  pub fn complement_signed(self, n: usize) -> (Sign, Self) {
    let complement = Self::from_bits(!self.bits() & Self::full(n).bits());
    let (sign, _) = self
      .union_signed(complement)
      .expect("Complement is disjoint.");
    (sign, complement)
  }

  /// Alternating single-element deletions $(-1)^i (S without s_i)$:
  /// the boundary of a simplex and the interior product of a blade.
  pub fn deletions(self) -> impl Iterator<Item = (Sign, usize, Self)> {
    self.iter().enumerate().map(move |(position, index)| {
      let deleted = Self::from_bits(self.bits() & !(1 << index));
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
