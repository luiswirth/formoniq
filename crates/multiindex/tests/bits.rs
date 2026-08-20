//! The bitset backing, at every width.

use multiindex::bits::{Bits, set_bits};

/// Numeric order on the bitsets is the order the crate's ranking convention
/// rests on, and it holds at every width.
#[test]
fn low_mask_and_singletons_agree_with_the_symbols() {
  fn check<B: Bits>() {
    assert!(B::low_mask(0).is_empty());
    assert_eq!(B::low_mask(B::WIDTH), B::MAX);
    assert_eq!(B::MAX.count_ones(), B::WIDTH);
    assert_eq!(B::ZERO.trailing_zeros(), B::WIDTH);
    for bit in 0..B::WIDTH {
      assert_eq!(B::singleton(bit).count_ones(), 1);
      assert_eq!(B::singleton(bit).trailing_zeros(), bit);
      assert_eq!(B::low_mask(bit).count_ones(), bit);
      assert_eq!(
        set_bits(B::low_mask(bit)).collect::<Vec<_>>(),
        (0..bit).collect::<Vec<_>>()
      );
    }
  }
  check::<u8>();
  check::<u16>();
  check::<u32>();
  check::<u64>();
  check::<u128>();
}

/// A right shift past the width empties the set rather than trapping, which
/// is what lets the colex successor run without a width special case.
#[test]
fn the_total_shift_empties_rather_than_trapping() {
  fn check<B: Bits>() {
    assert!(B::MAX.shr_total(B::WIDTH).is_empty());
    assert!(B::MAX.shr_total(B::WIDTH + 7).is_empty());
    assert_eq!(B::MAX.shr_total(0), B::MAX);
  }
  check::<u8>();
  check::<u128>();
}
