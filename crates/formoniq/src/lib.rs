#![doc = include_str!("../README.md")]

extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

pub mod assemble;
pub mod bc;
pub mod fe;
pub mod hx;
pub mod linalg;
pub mod matfree;
pub mod multigrid;
pub mod operators;
pub mod problems;
pub mod time;
pub mod trimmed_complex;
pub mod whitney_complex;
