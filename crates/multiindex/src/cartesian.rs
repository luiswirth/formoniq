//! Cartesian multi-indices: positional (radix) numbers.
//!
//! Elements of the product ${0, dots, "radix"-1}^d$, the index sets of
//! tensor-product structures. A cartesian index with radix 2 is exactly a
//! subset of the axes: the corners of the $d$-cube are [`Combination`]s,
//! and the Kuhn triangulation of the cube consists of the maximal chains
//! $emptyset subset {a_1} subset {a_1, a_2} subset dots.c$ in this subset
//! lattice, one for each permutation of the axes.

use super::Combination;

/// Converts a linear index in `0..radix^dim` to a cartesian multi-index in
/// ${0, dots, "radix"-1}^"dim"$ (least significant axis first).
pub fn linear_index2cartesian_index(mut lin_idx: usize, radix: usize, dim: usize) -> Vec<usize> {
  let mut cart_idx = vec![0; dim];
  for icomp in cart_idx.iter_mut() {
    *icomp = lin_idx % radix;
    lin_idx /= radix;
  }
  cart_idx
}

/// Converts a cartesian multi-index in ${0, dots, "radix"-1}^"dim"$ to a
/// linear index in `0..radix^dim`.
pub fn cartesian_index2linear_index(cart_idx: &[usize], radix: usize) -> usize {
  let mut lin_idx = 0;
  for &icomp in cart_idx.iter().rev() {
    lin_idx *= radix;
    lin_idx += icomp;
  }
  lin_idx
}

/// The linear-index offset of a cube corner (a set of axes with coordinate 1)
/// under the given per-axis strides.
pub fn corner_offset(corner: Combination, strides: &[usize]) -> usize {
  corner.iter().map(|axis| strides[axis]).sum()
}

/// The per-axis strides of the linear index of a cartesian grid:
/// $"stride"_i = "radix"^i$.
pub fn strides(radix: usize, dim: usize) -> Vec<usize> {
  (0..dim)
    .scan(1, |stride, _| {
      let this = *stride;
      *stride *= radix;
      Some(this)
    })
    .collect()
}
