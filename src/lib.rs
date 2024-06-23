extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

pub mod assemble;
pub mod fe;
pub mod geometry;
pub mod mesh;
pub mod orientation;
pub mod space;
pub mod util;

pub type Dim = usize;
