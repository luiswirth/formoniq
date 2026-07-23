//! `formoniq-studio`: the visual, interactive counterpart to `formoniq` -- a
//! viewer for inspecting PDE solutions, meshes and simplicial manifolds,
//! cochains, and the differential geometry underneath them. See
//! `crates/studio/CLAUDE.md` for the intrinsic/extrinsic seam this crate lives
//! on either side of.

extern crate nalgebra as na;

pub mod app;
pub mod demos;
pub(crate) mod display;
/// Headless PNG/MP4 rendering. Native only: it writes files and pipes to
/// `ffmpeg`, neither of which the browser offers.
#[cfg(not(target_arch = "wasm32"))]
pub mod export;
pub mod gallery;
pub mod render;
pub mod scene;
pub(crate) mod solve;
pub mod ui;
pub(crate) mod welcome;

/// The web entry point and its platform glue (canvas mount, async device
/// bootstrap, console logging). Isolated here so nothing web-specific leaks
/// into the shared viewer code.
#[cfg(target_arch = "wasm32")]
mod web;

#[cfg(not(target_arch = "wasm32"))]
pub use app::run;
