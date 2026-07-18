//! Linear-algebra backends and the type aliases the rest of the workspace
//! builds on: dense and sparse nalgebra types, a faer bridge for sparse
//! solves, and shift-invert eigensolving.
//!
//! Backends by role: nalgebra dense for element-local math, nalgebra-sparse
//! for globally assembled operators, faer for solves and eigenproblems. This
//! crate is where all three meet, so that no crate above it has to choose a
//! backend on its own.

extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

pub mod eigen;
pub mod faer;
pub mod nalgebra;
