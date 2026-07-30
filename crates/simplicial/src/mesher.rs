//! Mesh generators, combinatorial half.
//!
//! A generator's output is a topology and usually a geometry, and the two
//! separate here as everywhere else. The Kuhn triangulation of a box is pure
//! combinatorics: which simplices, in which vertex order, from the per-axis cell
//! counts alone. Placing the vertices in space is `regge`'s.
pub mod grid;
