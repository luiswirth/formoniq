#![allow(clippy::len_without_is_empty)]

extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

pub mod problems;

pub mod assemble;
pub mod exterior;
pub mod fe;
pub mod mesh;

pub mod combo;
pub mod linalg;
pub mod sparse;
pub mod util;

pub type Dim = usize;
pub type Codim = usize;
