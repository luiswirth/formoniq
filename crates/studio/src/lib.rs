//! `formoniq-studio`: the visual, interactive counterpart to `formoniq` -- a
//! viewer for inspecting PDE solutions, meshes and simplicial manifolds,
//! cochains, and the differential geometry underneath them. Runs natively and
//! on the web from one source; see `crates/studio/CLAUDE.md` for the
//! intrinsic/extrinsic seam this crate lives on either side of.

extern crate nalgebra as na;

pub mod app;
pub mod bake;
pub mod demos;
pub mod gallery;
pub mod io;
pub mod render;
pub mod scene;
pub mod streamline;
pub mod ui;

pub use app::run;
