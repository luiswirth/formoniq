//! The solve machinery: a faer bridge for sparse LU/Cholesky, shift-invert
//! eigensolving, and the sparse bilinear-form helpers assembly needs on top of
//! `simplicial`'s matrix types.
//!
//! `formoniq` is the only crate in the workspace that actually solves a linear
//! system, so this is where faer lives -- not a shared base crate every leaf
//! would otherwise compile it for free.

use simplicial::linalg::{CsrMatrix, Vector};

pub mod eigen;
pub mod faer;

pub fn bilinear_form_sparse(mat: &CsrMatrix, u: &Vector, v: &Vector) -> f64 {
  ((mat.transpose() * u).transpose() * v).x
}
pub fn quadratic_form_sparse(mat: &CsrMatrix, u: &Vector) -> f64 {
  bilinear_form_sparse(mat, u, u)
}
