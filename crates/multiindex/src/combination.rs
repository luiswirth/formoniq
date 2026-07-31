//! Subsets: the multi-indices of the exterior algebra.

use crate::{
  Sign, binomial,
  bits::{self, Bits},
  monotone::{DefaultBits, MonoIndexOver, Repetition},
};

/// A strictly increasing multi-index: a finite set of indices, stored as the
/// bitset of its symbols.
///
/// The basis element of $Lambda^k$ (a set of covector indices) and of
/// simplicial chains (a set of vertices).
///
/// A newtype over [`MonoIndexOver`] at [`Repetition::Forbidden`]: a subset is a
/// monotone word that may not repeat, and forbidding repetition makes the shift
/// zero, so the shifted word a `MonoIndex` stores is the set. The wrapper adds
/// nothing to the representation and delegates every operation; what it adds is
/// the family, so a multiset cannot be handed to a subset's operations, and the
/// set-theoretic vocabulary ([`Self::contains`], [`Self::is_subset_of`],
/// [`Self::subsets`]) that a multiset does not carry.
///
/// The derived `Ord` compares the bitsets numerically, which for equal
/// cardinality is exactly the colexicographic order.
///
/// The backing width is the type parameter, as on [`MonoIndexOver`], and
/// [`Combination`] is this at the width the workspace reads.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CombinationOver<B: Bits = DefaultBits>(MonoIndexOver<B>);

/// A subset at the default width.
pub type Combination = CombinationOver<DefaultBits>;

impl<B: Bits> CombinationOver<B> {
  pub fn empty() -> Self {
    Self(MonoIndexOver::empty(Repetition::Forbidden))
  }
  pub fn single(index: usize) -> Self {
    Self(MonoIndexOver::single(Repetition::Forbidden, index))
  }
  /// The full set ${0, dots, card - 1}$.
  pub fn full(card: usize) -> Self {
    Self::from_bits(B::low_mask(card))
  }
  /// From strictly increasing indices.
  pub fn from_increasing(indices: impl IntoIterator<Item = usize>) -> Self {
    Self(MonoIndexOver::new(Repetition::Forbidden, indices))
  }

  /// Canonicalize an arbitrarily ordered index word into the sign of its
  /// permutation and the underlying combination.
  ///
  /// `None` if an index repeats. The only place unsorted multi-indices
  /// exist: as transient inputs.
  pub fn from_word(word: impl IntoIterator<Item = usize>) -> Option<(Sign, Self)> {
    MonoIndexOver::from_word(Repetition::Forbidden, word).map(|(sign, index)| (sign, Self(index)))
  }

  /// The raw bitset, each set bit an index.
  pub fn bits(self) -> B {
    self.0.shifted_bits()
  }
  /// From a raw bitset, each set bit an index.
  ///
  /// Every bitset denotes a combination, so this is total. The name marks that
  /// the caller is working at the bit level, where the set is the bits.
  pub fn from_bits(bits: B) -> Self {
    Self(MonoIndexOver::from_shifted(Repetition::Forbidden, bits))
  }
  pub fn card(self) -> usize {
    self.0.degree()
  }
  pub fn is_empty(self) -> bool {
    self.card() == 0
  }
  pub fn contains(self, index: usize) -> bool {
    index < B::WIDTH && !(self.bits() & B::singleton(index)).is_empty()
  }
  pub fn is_subset_of(self, other: Self) -> bool {
    self.bits() & other.bits() == self.bits()
  }

  /// The indices in ascending order.
  pub fn iter(self) -> impl Iterator<Item = usize> {
    bits::set_bits(self.bits())
  }
  /// The position-th smallest index.
  pub fn index_at(self, position: usize) -> usize {
    self.iter().nth(position).expect("Position out of bounds.")
  }
  /// With the index added; must not be contained yet.
  pub fn inserted(self, index: usize) -> Self {
    assert!(!self.contains(index));
    Self::from_bits(self.bits() | B::singleton(index))
  }
  /// The position of an index within the set.
  pub fn position_of(self, index: usize) -> usize {
    assert!(self.contains(index));
    (self.bits() & B::low_mask(index)).count_ones()
  }

  /// Colexicographic rank among all combinations of the same cardinality:
  /// the combinatorial number system $sum_i binom(s_i, i+1)$.
  ///
  /// Independent of any ambient dimension. Ranks of combinations inside
  /// ${0, dots, n-1}$ are exactly $0..binom(n, "card")$.
  pub fn rank(self) -> usize {
    self.0.rank()
  }
  /// Inverse of [`Self::rank`]: greedy from the largest element.
  pub fn from_rank(card: usize, rank: usize) -> Self {
    Self(MonoIndexOver::from_rank(Repetition::Forbidden, card, rank))
  }

  /// All combinations of the given cardinality in colexicographic order.
  ///
  /// The universal enumeration: take the first $binom(n, "card")$ to get
  /// exactly the combinations inside ${0, dots, n-1}$.
  pub fn all(card: usize) -> impl Iterator<Item = Self> {
    let mut next = Some(Self::full(card));
    std::iter::from_fn(move || {
      let current = next?;
      next = current.0.colex_successor().map(Self);
      Some(current)
    })
  }

  /// The combinations of the given cardinality inside ${0, dots, n-1}$, in
  /// colexicographic order: the prefix of [`Self::all`] the filtration
  /// property picks out.
  pub fn inside(n: usize, card: usize) -> impl Iterator<Item = Self> {
    Self::all(card).take(binomial(n, card))
  }

  /// Merge two disjoint combinations with the sign of the interleaving
  /// permutation: the wedge of basis blades. `None` if they intersect.
  pub fn union_signed(self, other: Self) -> Option<(Sign, Self)> {
    self
      .0
      .merge(&other.0)
      .map(|(sign, union)| (sign, Self(union)))
  }

  /// The complement within ${0, dots, n-1}$ and the sign such that
  /// $e_S wedge e_(S^c) = "sign" dot e_({0, dots, n-1})$: the combinatorics
  /// of the Hodge star.
  pub fn complement_signed(self, n: usize) -> (Sign, Self) {
    let (sign, complement) = self.0.complement_signed(n);
    (sign, Self(complement))
  }

  /// Alternating single-element deletions $(-1)^i (S without s_i)$:
  /// the boundary of a simplex and the interior product of a blade.
  pub fn deletions(self) -> impl Iterator<Item = (Sign, usize, Self)> {
    self
      .0
      .deletions()
      .map(|(sign, index, deleted)| (sign, index, Self(deleted)))
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
  Combination::inside(n, card)
}

impl<B: Bits> FromIterator<usize> for CombinationOver<B> {
  /// From strictly increasing indices.
  fn from_iter<T: IntoIterator<Item = usize>>(iter: T) -> Self {
    Self::from_increasing(iter)
  }
}

impl<B: Bits> From<CombinationOver<B>> for MonoIndexOver<B> {
  fn from(combination: CombinationOver<B>) -> Self {
    combination.0
  }
}

impl<B: Bits> std::fmt::Debug for CombinationOver<B> {
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
  ///
  /// Checked at several backings: the width bounds what a value can hold and
  /// enters neither the enumeration nor the rank, so the same statement has to
  /// come out of every one of them.
  #[test]
  fn colex_enumeration_is_filtration_compatible() {
    fn check<B: Bits>() {
      for n in 0..=6 {
        for card in 0..=n {
          let inside: Vec<_> = CombinationOver::<B>::inside(n, card).collect();
          assert_eq!(inside.len(), binomial(n, card));
          assert!(inside.iter().all(|c| c.iter().all(|index| index < n)));
          assert_eq!(
            inside,
            itertools::Itertools::combinations(0..n, card)
              .map(CombinationOver::<B>::from_increasing)
              .sorted()
              .collect::<Vec<_>>()
          );
        }
      }
    }
    check::<u8>();
    check::<u16>();
    check::<u64>();
    check::<u128>();
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
    fn check<B: Bits>() {
      for n in 0..=6 {
        for card in 0..=n {
          for combination in CombinationOver::<B>::inside(n, card) {
            let (sign, complement) = combination.complement_signed(n);
            let (union_sign, union) = combination.union_signed(complement).unwrap();
            assert_eq!(union, CombinationOver::<B>::full(n));
            assert_eq!(sign, union_sign);
          }
        }
      }
    }
    check::<u8>();
    check::<u16>();
    check::<u64>();
    check::<u128>();
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
