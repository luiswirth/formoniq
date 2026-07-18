//! Combinatorics of finite index sets.
//!
//! The central type is [`Combination`]: a strictly increasing multi-index
//! stored as a bitset. It is the canonical basis element of both simplicial
//! chains (as a set of vertices) and exterior algebra blades (as a set of
//! covector indices). Arbitrarily ordered index words exist only transiently:
//! [`Combination::from_word`] canonicalizes them into a [`Sign`] and a
//! [`Combination`].
//!
//! All ranks and enumerations are **colexicographic**: combinations are
//! compared by their largest element first, which coincides with the numeric
//! order of the bitsets. The rank is the combinatorial number system
//! $"rank"{s_0 < dots.c < s_(k-1)} = sum_i binom(s_i, i+1)$,
//! independent of any ambient dimension, and the first $binom(n,k)$
//! combinations of the universal enumeration are exactly those inside
//! ${0, dots, n-1}$.

pub mod cartesian;

/// The dimension of a space or object.
pub type Dim = usize;

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

/// All weak compositions of `total` into `parts` nonnegative parts:
/// the tuples $k in NN_0^p$ with $sum_i k_i = "total"$, of which there are
/// $binom("total" + p - 1, p - 1)$.
///
/// The multiset sibling of [`Combination`], and enumerated as one: a
/// composition is $p - 1$ bars among $"total" + p - 1$ slots (stars and bars),
/// so this is [`combinations`] read through that bijection. The order is the
/// colex order of the bars, and it is therefore the same convention the rest of
/// the crate indexes by.
///
/// Total at both ends: one part admits only $("total")$, and no parts admit
/// only the empty composition of zero.
pub fn compositions(parts: usize, total: usize) -> impl Iterator<Item = Vec<usize>> {
  let bars = parts.saturating_sub(1);
  let slots = total + bars;
  let count = if parts == 0 {
    usize::from(total == 0)
  } else {
    binomial(slots, bars)
  };
  combinations(slots, bars).take(count).map(move |bar_set| {
    if parts == 0 {
      return Vec::new();
    }
    // The gaps the bars cut the slots into, reversed so that the leading part
    // is the last gap: that is what makes `total = 1` list the parts in the
    // order the standard basis places the vertices.
    let mut composition = Vec::with_capacity(bars + 1);
    let mut previous = None;
    for bar in bar_set.iter() {
      composition.push(bar - previous.map_or(0, |p| p + 1));
      previous = Some(bar);
    }
    composition.push(slots - previous.map_or(0, |p| p + 1));
    composition.reverse();
    composition
  })
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

  /// Stars and bars: $binom("total" + p - 1, p - 1)$ distinct tuples summing to
  /// `total`. Total at the degenerate end, where no parts can compose only zero.
  #[test]
  fn compositions_are_stars_and_bars() {
    for parts in 0..=5 {
      for total in 0..=5 {
        let all: Vec<_> = compositions(parts, total).collect();
        let expected = if parts == 0 {
          usize::from(total == 0)
        } else {
          binomial(total + parts - 1, parts - 1)
        };
        assert_eq!(all.len(), expected);
        assert_eq!(all.iter().unique().count(), all.len());
        for composition in &all {
          assert_eq!(composition.len(), parts);
          assert_eq!(composition.iter().sum::<usize>(), total);
        }
      }
    }
  }

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
