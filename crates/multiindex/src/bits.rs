//! The backing integer of a bitset, as a trait.
//!
//! A monotone multi-index is the bitset of its shifted word, and the width of
//! that bitset is a fact about the machine rather than about the mathematics:
//! it bounds the alphabet a *representation* can hold, and nothing else. So it
//! is a type parameter, and [`Bits`] is what a backing type has to supply for
//! the index operations to be written once over every width.
//!
//! The trait carries the bit and shift operators, the two population primitives
//! and the one carrying addition Gosper's hack needs, and stops there. It is
//! deliberately narrower than an integer: an index bitset is not a number, and a
//! bound admitting general arithmetic on one would say something false about
//! what the code depends on.
//!
//! It is sealed, and over the primitive unsigned integers alone. The reason is
//! the ordering: comparing two bitsets numerically *is* colexicographic order
//! at equal cardinality, which is the crate's indexing convention, and it is
//! exactly what a `Vec`-backed bitset would lose. Sealing makes that a compile
//! error rather than a comment.

use std::{
  fmt::Debug,
  hash::Hash,
  ops::{BitAnd, BitOr, BitXor, Not, Shl, Shr},
};

mod sealed {
  pub trait Sealed {}
}

/// The backing integer of a bitset of symbols.
///
/// `WIDTH` is the number of symbols the set can range over, so it is the
/// ceiling on a [`MonoIndex`](crate::MonoIndex)'s *shifted* alphabet: an
/// alternating index may span that many symbols, and a symmetric one over $n$
/// symbols reaches degree $"WIDTH" - n + 1$.
///
/// Implemented for `u8` through `u128` and sealed against anything else.
pub trait Bits:
  sealed::Sealed
  + Copy
  + Default
  + Ord
  + Hash
  + Debug
  + BitAnd<Output = Self>
  + BitOr<Output = Self>
  + BitXor<Output = Self>
  + Not<Output = Self>
  + Shl<usize, Output = Self>
  + Shr<usize, Output = Self>
{
  /// The number of bits, hence the number of symbols representable.
  const WIDTH: usize;
  /// The empty set.
  const ZERO: Self;
  /// The singleton ${0}$, which shifted left by $i$ is the singleton ${i}$.
  const ONE: Self;
  /// The full set ${0, dots, "WIDTH" - 1}$.
  const MAX: Self;

  /// The cardinality of the set.
  fn count_ones(self) -> usize;
  /// The smallest element, and `WIDTH` on the empty set.
  fn trailing_zeros(self) -> usize;
  /// Addition, `None` on overflow.
  ///
  /// The one arithmetic operation the bit vocabulary does not cover: the colex
  /// successor carries a run of set bits upward, and the carry running off the
  /// top is what marks the last index of a width.
  fn checked_add(self, other: Self) -> Option<Self>;

  /// The set ${0, dots, n-1}$, for any `n` up to and including [`Self::WIDTH`].
  ///
  /// # Panics
  /// If `n` exceeds the width.
  fn low_mask(n: usize) -> Self {
    assert!(n <= Self::WIDTH, "alphabet exceeds the bitset");
    if n == Self::WIDTH {
      Self::MAX
    } else {
      !(Self::MAX << n)
    }
  }

  /// The singleton ${bit}$.
  ///
  /// # Panics
  /// If the bit lies past the width.
  fn singleton(bit: usize) -> Self {
    assert!(bit < Self::WIDTH, "index reaches past the bitset");
    Self::ONE << bit
  }

  /// Whether the set is empty.
  fn is_empty(self) -> bool {
    self == Self::ZERO
  }
  /// The set with its smallest element removed, and the empty set unchanged.
  fn without_lowest(self) -> Self {
    self ^ self.lowest()
  }
  /// The singleton of the smallest element, and the empty set unchanged.
  fn lowest(self) -> Self {
    if self.is_empty() {
      Self::ZERO
    } else {
      Self::ONE << self.trailing_zeros()
    }
  }
  /// A right shift that is total in the shift amount, emptying the set once
  /// nothing can survive it.
  fn shr_total(self, amount: usize) -> Self {
    if amount >= Self::WIDTH {
      Self::ZERO
    } else {
      self >> amount
    }
  }
  /// A left shift that is total in the shift amount.
  ///
  /// # Panics
  /// If a set element would be shifted past the width, which is the same
  /// contract as [`Self::singleton`]: the empty set shifts anywhere.
  fn shl_total(self, amount: usize) -> Self {
    if amount >= Self::WIDTH {
      assert!(self.is_empty(), "index reaches past the bitset");
      Self::ZERO
    } else {
      self << amount
    }
  }
}

/// The widest backing a [`Bits`] can have.
///
/// Not a bound on any one index, which is [`Bits::WIDTH`], but on the family:
/// what a table indexed by a shifted symbol has to cover to serve every width.
pub const MAX_WIDTH: usize = 128;

macro_rules! impl_bits {
  ($($ty:ty),*) => {$(
    impl sealed::Sealed for $ty {}
    impl Bits for $ty {
      const WIDTH: usize = <$ty>::BITS as usize;
      const ZERO: Self = 0;
      const ONE: Self = 1;
      const MAX: Self = <$ty>::MAX;
      fn count_ones(self) -> usize {
        <$ty>::count_ones(self) as usize
      }
      fn trailing_zeros(self) -> usize {
        <$ty>::trailing_zeros(self) as usize
      }
      fn checked_add(self, other: Self) -> Option<Self> {
        <$ty>::checked_add(self, other)
      }
    }
  )*};
}

impl_bits!(u8, u16, u32, u64, u128);

/// The set bits, ascending.
pub fn set_bits<B: Bits>(mut bits: B) -> impl Iterator<Item = usize> {
  std::iter::from_fn(move || {
    (!bits.is_empty()).then(|| {
      let symbol = bits.trailing_zeros();
      bits = bits.without_lowest();
      symbol
    })
  })
}

#[cfg(test)]
mod test {
  use super::*;

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
}
