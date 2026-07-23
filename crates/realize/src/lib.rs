//! `formoniq-realize`: the point at which intrinsic data becomes extrinsic.
//!
//! The engine below is intrinsic by discipline: a `Complex` is combinatorics, a
//! geometry is edge lengths, a field is a cochain, and none of it has a
//! position, a dimension a screen can show, or a component a file can name.
//! Something has to spend all three, and this crate is where that happens,
//! once, for every consumer.
//!
//! Two reductions carry it, and they are the same move made on the two axes:
//!
//! - **Dimension reduces to a drawable primitive** ([`surface`], [`bake`]): an
//!   $n$-manifold to $min(n, 2)$ -- a surface to wound triangles, a curve to
//!   segments, a solid to the 2-simplices of its boundary -- realized in $RR^3$
//!   through an embedding.
//! - **Grade reduces to a mark** ([`reduce`]): a $k$-form to its reduced grade
//!   $min(k, n-k)$ through the Hodge star, a scalar density at 0 and a tangent
//!   line field at 1.
//!
//! The two compose, and **the order is fixed: dimension first, grade second**.
//! The object a mark is a mark *of* is the reduced surface, so the $n$ in
//! $min(k, n-k)$ is the surface's, never the mesh's. A 2-form on a solid
//! reduces to arrows against the volume but to a density against the boundary,
//! and only the latter is a claim about anything a viewer or a file can show:
//! a flux has no direction in the surface carrying it.
//!
//! **Nothing here draws.** There is no GPU, no window and no rasterizer in the
//! dependency graph, which is the crate's reason for existing separately: a
//! headless run writes a solution to disk without a renderer in the build, and
//! a `.vtu` for ParaView is not reached through a viewer. What consumes these
//! reductions -- the [`io`] exporters here, the renderer above -- are peers,
//! and because they share the reduction rather than each carrying their own,
//! a disagreement between what the viewer draws and what an external tool
//! draws is a bug in one place instead of a drift between two.

extern crate nalgebra as na;

pub mod advect;
pub mod bake;
pub mod demos;
pub mod deposit;
pub mod glyph;
pub mod io;
pub mod reduce;
pub mod surface;
pub mod volume;
