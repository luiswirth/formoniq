//! The continuum manifold $M$: the smooth object a mesh approximates.
//!
//! There are two manifolds in this library, and they are independent objects.
//! One is the *simplicial* manifold $M_h$ -- piecewise-affine, combinatorial,
//! always present -- which is what the `simplicial` crate is. The other is the
//! *continuum* $M$: the smooth manifold $M_h$ approximates, analytic and exact,
//! existing only when it is given (by a parametrization, a level set, a
//! formula). A Regge mesh has no continuum, a continuum has no mesh, and neither
//! depends on the other; the mesh *approximates* the manifold, but that is a
//! relation between them, not a dependency of one on the other.
//!
//! This crate owns $M$ and everything analytic about it:
//!
//! - [`field`]: the mesh-independent analytic data *on* $M$ -- an exact
//!   solution, a source, a boundary flux -- as a [`CoordField`](field::CoordField)
//!   of exterior elements over a coordinate domain. This is *not* the
//!   discrete-differential-form notion of a field ([`Section`] lives in `derham`,
//!   over the simplicial manifold); it is a function of a point of the
//!   continuum's chart domain, valued in the flat exterior algebra.
//! - [`parametrization`]: the smooth [`Parametrization`](parametrization::Parametrization)
//!   $phi: Omega -> RR^N$ of $M$, and the chart $chi = phi^(-1) compose r$ it
//!   induces by orthogonal (nearest-point) projection.
//!
//! `chartan` depends on `exterior` (the flat algebra its fields are valued in)
//! and on nothing meshy: it is a *sibling* of `simplicial`, not a layer above or
//! below it. Their one relation -- pulling continuum data onto the simplicial
//! mesh, and the approximation error that costs -- is the join, and it lives one
//! crate up in `derham`, where `exterior`, `simplicial` and `chartan` meet.
//!
//! [`Section`]: https://docs.rs/derham

extern crate nalgebra as na;

pub mod field;
pub mod parametrization;
