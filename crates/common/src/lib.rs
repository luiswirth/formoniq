extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

pub mod affine;
pub mod combo;
pub mod coord;
pub mod gramian;

pub mod linalg;
pub mod util;

/// The dimension of a space or object.
pub type Dim = usize;
