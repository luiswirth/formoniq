//! The Whitney interpolation $W: C^k -> L^2 Lambda^k$.
//!
//! The lowest-order finite element interpolation of cochains into the trimmed
//! space $P^-_1 Lambda^k$ of differential forms affine on each cell. Together
//! with the de Rham map $R: L^2 Lambda^k -> C^k$ (see [`crate::derham`]) it
//! forms the pair of cochain maps at the heart of FEEC, governed by the
//! executable laws
//!
//! - $R compose W = id$: Whitney's theorem
//!   (test `whitney_basis_property` in [`crate`]).
//! - $dif compose W = W compose dif$: the Whitney forms are a subcomplex of
//!   the de Rham complex (test `whitney_interpolation_is_cochain_map` in
//!   [`interpolant`]).
//! - $R compose dif = dif compose R$: Stokes' theorem
//!   (test `derham_map_is_cochain_map` in [`crate::derham`]).

pub mod form;
pub mod interpolant;
