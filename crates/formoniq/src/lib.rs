#![doc = include_str!("../README.md")]

extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

pub mod bc;
pub mod fe;
pub mod galerkin;
pub mod harmonic;
pub mod hodge;
pub mod hx;
pub mod linalg;
pub mod matfree;
pub mod multigrid;
pub mod operators;
pub mod problems;
pub mod time;
pub mod whitney_complex;
