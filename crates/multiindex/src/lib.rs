#![doc = include_str!("../README.md")]

pub mod bits;
pub mod cartesian;
pub mod combination;
pub mod composition;
pub mod count;
pub mod degree;
pub mod monotone;
pub mod multi;
pub mod permutation;
pub mod sign;

pub use bits::Bits;
pub use cartesian::{Radix, Word, WordDeletions, Words};
pub use combination::{Combination, CombinationOver, combinations};
pub use composition::Composition;
pub use count::{binomial, factorial, factorial_f64};
pub use degree::{Degree, Dim};
pub use monotone::{
  DefaultBits, MonoDeletions, MonoDeletionsOver, MonoIndex, MonoIndexOver, MonoIndices,
  MonoIndicesOver, Repetition, Symbols,
};
pub use multi::{MultiDeletions, MultiIndex, MultiIndices};
pub use permutation::Permutation;
pub use sign::{Sign, sort_count_swaps, sort_signed};
