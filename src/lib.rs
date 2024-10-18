extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

pub mod assemble;
pub mod combinatorics;
pub mod fe;
pub mod geometry;
pub mod matrix;
pub mod mesh;
pub mod orientation;
pub mod space;
pub mod util;

pub type Dim = usize;
pub type Codim = usize;
