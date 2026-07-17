//! `formoniq-studio`: the visual, interactive counterpart to `formoniq` -- a
//! viewer for inspecting PDE solutions, meshes and simplicial manifolds,
//! cochains, and the differential geometry underneath them. See
//! `crates/studio/CLAUDE.md` for the intrinsic/extrinsic seam this crate lives
//! on either side of.

extern crate nalgebra as na;

pub mod advect;
pub mod app;
pub mod bake;
pub mod demos;
pub(crate) mod deposit;
pub(crate) mod display;
pub mod export;
pub mod gallery;
pub(crate) mod glyph;
pub mod io;
pub mod render;
pub mod scene;
pub mod ui;

pub use app::run;
