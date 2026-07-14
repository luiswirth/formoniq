//! The extrinsic layer: a coordinate realization of the mesh.
//!
//! An embedding is *one* geometry among several (invariant 2), and everything
//! here is downstream of it: vertex coordinates, the affine parametrization of a
//! cell, point location. A manifold given by Regge edge lengths has none of it,
//! and the core FEEC path must never ask for it.
//!
//! Ambient coordinates are therefore kept apart, by type, from the
//! [`Bary`](crate::atlas::Bary) and [`Local`](crate::atlas::Local) coordinates
//! of a chart: those are intrinsic and exist on every geometry, and the maps
//! between the two worlds are the parametrizations in
//! [`simplex`].

pub mod locate;
pub mod mesh;
pub mod simplex;

pub use common::coord::{Ambient, Coord, CoordRef};

use common::linalg::nalgebra::{RowVector, RowVectorView, Vector, VectorView};

pub type TangentVector = Vector;
pub type TangentVectorRef<'a> = VectorView<'a>;

pub type CoTangentVector = RowVector;
pub type CoTangentVectorRef<'a> = RowVectorView<'a>;
