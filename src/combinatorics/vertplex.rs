//! # Vertplex: A Combinatorial Representation of Simplices
//!
//! The `vertplex` module provides abstractions for working with simplices in a
//! purely combinatorial manner.
//!
//! A **vertplex** represents a simplex as a collection of vertices, entirely
//! independent of any geometry. It captures the combinatorial structure of a
//! simplex, focusing only on its vertex set and relationships, such as subsets
//! and orientations.
//!
//! ## Types of Vertplexes
//!
//! - [`CanonicalVertplex`]: A simplex with vertices sorted in canonical order,
//!   enabling efficient comparison of vertex sets and subset relationships.
//! - [`OrderedVertplex`]: A simplex with vertices ordered explicitly,
//!   preserving the sequence as provided.
//! - [`OrientedVertplex`]: A simplex with an explicit orientation symbol, combining the orientation given by the vertex ordering
//!   with an additional orientation symbol.

mod canonical;
mod ordered;
mod oriented;

pub use canonical::CanonicalVertplex;
pub use ordered::OrderedVertplex;
pub use oriented::OrientedVertplex;
