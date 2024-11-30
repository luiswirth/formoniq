extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

pub mod assemble;
pub mod cell;
pub mod combinatorics;
pub mod fe;
pub mod mesh;
pub mod space;

pub mod evp;
pub mod lse;

pub mod matrix;
pub mod util;

pub type Dim = usize;
pub type Codim = usize;

pub type VertexIdx = usize;
