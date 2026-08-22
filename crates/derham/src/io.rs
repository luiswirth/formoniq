//! Interchange formats for a simplicial manifold carrying discrete forms.
//!
//! A format wants an embedding and a form already reduced to a scalar or a
//! vector, so everything the crates below keep intrinsic has to be spent
//! before a file can be written. That is the boundary of the API invariant 2
//! draws, and it is where these live.

pub mod vtu;
