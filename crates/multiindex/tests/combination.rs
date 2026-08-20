//! Colexicographic order and the combinatorics of [`Combination`].

use itertools::Itertools;
use multiindex::{Bits, Combination, CombinationOver, Sign, binomial, combinations};

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
