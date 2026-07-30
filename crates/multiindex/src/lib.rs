#![doc = include_str!("../README.md")]

pub mod cartesian;
pub mod combination;
pub mod composition;
pub mod count;
pub mod degree;
pub mod monotone;
pub mod multi;
pub mod permutation;
pub mod sign;

pub use cartesian::{Radix, Word, WordDeletions, Words};
pub use combination::{Combination, MAX_NINDICES, combinations};
pub use composition::Composition;
pub use count::{binomial, factorial, factorial_f64};
pub use degree::{Degree, Dim};
pub use monotone::{MonoDeletions, MonoIndex, MonoIndices, Repetition, Symbols};
pub use multi::{MultiDeletions, MultiIndex, MultiIndices};
pub use permutation::Permutation;
pub use sign::{Sign, sort_count_swaps, sort_signed};
