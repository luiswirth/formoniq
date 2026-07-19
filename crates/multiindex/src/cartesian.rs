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
pub fn linear2cartesian(mut lin_idx: usize, radix: usize, dim: usize) -> Vec<usize> {
  let mut cart_idx = vec![0; dim];
  for icomp in cart_idx.iter_mut() {
    *icomp = lin_idx % radix;
    lin_idx /= radix;
  }
  cart_idx
}

/// Converts a cartesian multi-index in ${0, dots, "radix"-1}^"dim"$ to a
/// linear index in `0..radix^dim`.
pub fn cartesian2linear(cart_idx: &[usize], radix: usize) -> usize {
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

#[cfg(test)]
mod tests {
  use super::*;
  use crate::combinations;
  use itertools::Itertools;

  /// Linear and cartesian indexing are mutually inverse over the whole grid
  /// $0.."radix"^"dim"$, and every cartesian component stays in
  /// ${0, dots, "radix"-1}$. Includes the degenerate $"dim" = 0$ grid, whose
  /// single point is the empty tuple at linear index 0.
  #[test]
  fn linear_cartesian_round_trip() {
    for radix in 1usize..=4 {
      for dim in 0..=4 {
        let count = radix.pow(dim as u32);
        for lin in 0..count {
          let cart = linear2cartesian(lin, radix, dim);
          assert_eq!(cart.len(), dim);
          assert!(cart.iter().all(|&c| c < radix));
          assert_eq!(cartesian2linear(&cart, radix), lin);
        }
      }
    }
  }

  /// The strides are the radix powers $"radix"^i$, and the linear index is the
  /// stride-weighted sum of the cartesian components.
  #[test]
  fn strides_are_radix_powers_and_reconstruct_linear_index() {
    for radix in 1usize..=4 {
      for dim in 0..=4 {
        let strides = strides(radix, dim);
        assert_eq!(strides.len(), dim);
        for (i, &stride) in strides.iter().enumerate() {
          assert_eq!(stride, radix.pow(i as u32));
        }
        for lin in 0..radix.pow(dim as u32) {
          let cart = linear2cartesian(lin, radix, dim);
          let weighted: usize = cart.iter().zip(&strides).map(|(&c, &s)| c * s).sum();
          assert_eq!(weighted, lin);
        }
      }
    }
  }

  /// A cube corner is a radix-2 cartesian index: its stride offset equals the
  /// linear index of the 0/1 indicator vector of the chosen axes.
  #[test]
  fn corner_offset_is_the_indicator_linear_index() {
    for dim in 0..=4 {
      let strides = strides(2, dim);
      for card in 0..=dim {
        for corner in combinations(dim, card) {
          let indicator: Vec<usize> = (0..dim)
            .map(|axis| usize::from(corner.contains(axis)))
            .collect();
          assert_eq!(
            corner_offset(corner, &strides),
            cartesian2linear(&indicator, 2)
          );
        }
      }
    }
  }

  /// The Kuhn triangulation claim: each permutation of the axes gives the
  /// maximal chain $emptyset subset {a_0} subset {a_0, a_1} subset dots.c$ of
  /// cube corners. Consecutive corners differ by exactly one axis (so the
  /// simplex edge vectors are the standard basis vectors, unit volume $1/d!$),
  /// the chain has $"dim"+1$ corners ending at the full cube, and the added
  /// axes are a permutation of $0.."dim"$. There are $"dim"!$ such chains.
  #[test]
  fn kuhn_chains_are_maximal_and_cover() {
    for dim in 0..=4 {
      let mut chain_count = 0;
      for perm in (0..dim).permutations(dim) {
        chain_count += 1;
        let mut corner = Combination::empty();
        let mut added = Vec::new();
        let mut corners = vec![corner];
        for &axis in &perm {
          assert!(!corner.contains(axis));
          corner = corner.inserted(axis);
          added.push(axis);
          corners.push(corner);
        }
        assert_eq!(corners.len(), dim + 1);
        assert_eq!(corner, Combination::full(dim));
        added.sort_unstable();
        assert_eq!(added, (0..dim).collect::<Vec<_>>());
        // Nested chain: each corner a subset of the next.
        assert!(corners.windows(2).all(|w| w[0].is_subset_of(w[1])));
      }
      assert_eq!(chain_count, (1..=dim).product::<usize>());
    }
  }
}
