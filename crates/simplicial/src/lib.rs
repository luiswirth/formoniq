#![doc = include_str!("../README.md")]

extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

pub mod atlas;
pub mod geometry;
pub mod topology;

pub mod io;
pub mod linalg;
pub mod mesher;

/// [`Sign`] is re-exported because it is part of this crate's own surface: the
/// orientation of a simplex, the coefficients of a boundary, and a cell's
/// coherent orientation are all values of it, and a consumer must be able to
/// name what those methods return without depending on `multiindex` directly.
pub use multiindex::{Dim, Sign};
