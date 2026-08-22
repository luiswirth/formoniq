#![doc = include_str!("../README.md")]

extern crate nalgebra as na;

/// The (co)chain complex is re-exported because it is what this crate's maps
/// consume and return, and a reader must be able to name that without reaching
/// past this crate. A [`Cochain`] is a simplicial-topology object, one
/// coefficient per simplex; it becomes a *discrete differential form* only
/// through [`derham_map`](project::derham_map), which integrates a form over
/// each simplex and is the reading this crate adds.
pub use simplicial::topology::chain::{Chain, Cochain, pairing};

pub mod interpolate;
pub mod io;
pub mod project;
pub mod prolongate;
pub mod reduce;
pub mod section;
