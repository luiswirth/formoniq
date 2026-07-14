//! The Whitney interpolation $W: C^k -> L^2 Lambda^k$.
//!
//! The lowest-order finite element interpolation of cochains into piecewise
//! linear differential forms. Together with the de Rham map
//! $R: L^2 Lambda^k -> C^k$ (see [`crate::derham`]) it forms the pair of
//! cochain maps at the heart of FEEC, governed by the executable laws
//!
//! - $R compose W = id$: Whitney's theorem
//!   (test `whitney_basis_property` in [`crate`]).
//! - $dif compose W = W compose dif$: the Whitney forms are a subcomplex of
//!   the de Rham complex (test `whitney_interpolation_is_cochain_map` in
//!   [`form`]).
//! - $R compose dif = dif compose R$: Stokes' theorem
//!   (test `derham_map_is_cochain_map` in [`crate::derham`]).

pub mod form;
pub mod lsf;
