#![allow(clippy::len_without_is_empty)]

extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

pub mod assemble;
pub mod combo;
pub mod exterior;
pub mod fe;
pub mod geometry;
pub mod mesh;
pub mod simplicial;
pub mod space;

pub mod linalg;
pub mod sparse;
pub mod util;

pub mod lse;

pub mod helmholtz;
pub mod poisson;

pub mod diffusion;
pub mod wave;

pub type Dim = usize;
pub type Codim = usize;

pub type VertexIdx = usize;
