pub mod cartesian;
pub mod quotient;
pub mod quotient_embed;
pub mod sphere;
pub mod teaching;

use simplicial::linalg::Vector;

/// The per-axis cell counts making the spacing quasi-uniform: each axis gets a
/// count in proportion to its own extent, the longest of them
/// `ncells_longest`, so the cells come out as near cubical as integer counts
/// allow.
///
/// This is what keeps a long thin domain from being meshed into slivers. One
/// count over unequal extents reproduces the domain's own aspect ratio in every
/// cell, and the shape regularity that every FEM error constant depends on
/// degrades with it. Shape regularity is a property of the spacing, never of
/// the counts.
///
/// Every axis keeps at least one cell. A caller whose axes need more, as a
/// closed one of a quotient does, raises the floor itself.
pub fn quasi_uniform_counts(extents: &Vector, ncells_longest: usize) -> Vec<usize> {
  let longest = extents.iter().copied().fold(0.0_f64, f64::max);
  extents
    .iter()
    .map(|&extent| ((extent / longest * ncells_longest as f64).round() as usize).max(1))
    .collect()
}
