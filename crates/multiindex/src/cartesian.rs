//! Cartesian multi-indices: positional (radix) numbers.
//!
//! Elements of the product ${0, dots, "radix"-1}^d$, the index sets of
//! tensor-product structures, and so the basis of the free tensor power
//! $V^(times.circle d)$ itself. A cartesian index with radix 2 is exactly a
//! subset of the axes: the corners of the $d$-cube are [`Combination`]s,
//! and the Kuhn triangulation of the cube consists of the maximal chains
//! $emptyset subset {a_1} subset {a_1, a_2} subset dots.c$ in this subset
//! lattice, one for each permutation of the axes.
//!
//! This is the family with no symmetry to exploit, and the representation
//! says so. A monotone word is a set once shifted, hence a bitset with an
//! alphabet-independent rank. A cartesian index is neither, and cannot be.
//! There is no quotient here, so there is nothing to compress and no way to
//! number the basis without knowing how wide it is.

use super::Combination;
use crate::monotone::Symbols;

/// A word over an alphabet: a basis element of $V^(times.circle k)$.
///
/// Carries its own alphabet, unlike [`MonoIndex`](crate::MonoIndex), because a
/// radix rank cannot be taken without one. That asymmetry is forced: the colex
/// rank of a monotone index is a sum of binomials in its symbols alone, while
/// the rank of a word is a positional number and positions have a base.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Word {
  /// The word packed as its own radix rank, most significant position first.
  ///
  /// A word is a positional number, so storing the symbols separately would
  /// be storing the same thing twice. The packing costs no generality: a word
  /// indexes a component of $V^(times.circle k)$, so $n^k$ has to fit in a
  /// `usize` for that component to exist at all.
  rank: usize,
  degree: usize,
  radix: usize,
}

impl Word {
  /// A word over an alphabet of `radix` symbols.
  ///
  /// # Panics
  /// If any symbol is not below the radix.
  pub fn new(radix: usize, symbols: impl IntoIterator<Item = usize>) -> Self {
    let mut degree = 0;
    let mut rank = 0;
    for symbol in symbols {
      assert!(symbol < radix, "a word's symbols lie below its radix");
      rank = rank * radix + symbol;
      degree += 1;
    }
    Self {
      rank,
      degree,
      radix,
    }
  }

  /// The empty word: the basis of $V^(times.circle 0) = RR$.
  pub fn empty(radix: usize) -> Self {
    Self {
      rank: 0,
      degree: 0,
      radix,
    }
  }

  /// The number of positions, hence the degree of the power it indexes.
  pub fn degree(&self) -> usize {
    self.degree
  }
  pub fn radix(&self) -> usize {
    self.radix
  }
  /// The symbols, in position order, unpacked from the rank.
  pub fn symbols(&self) -> Symbols {
    let mut symbols = Symbols::from_iter(std::iter::repeat_n(0, self.degree));
    let mut rank = self.rank;
    for symbol in symbols.iter_mut().rev() {
      *symbol = rank % self.radix;
      rank /= self.radix;
    }
    symbols
  }
  pub fn symbol(&self, position: usize) -> usize {
    let from_the_end = self.degree - 1 - position;
    (self.rank / self.radix.pow(from_the_end as u32)) % self.radix
  }
  pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
    (0..self.degree).map(move |position| self.symbol(position))
  }

  /// The number of words of a degree over an alphabet: $n^k$.
  ///
  /// The dimension of $V^(times.circle k)$, and the count no symmetry reduces.
  pub fn count(radix: usize, degree: usize) -> usize {
    radix.pow(degree as u32)
  }

  /// The radix rank, last position running fastest:
  /// $sum_j w_j space n^(k-1-j)$.
  ///
  /// Depends on the alphabet, necessarily. The monotone families rank
  /// independently of it, so widening the ambient space renumbers nothing
  /// there; here it renumbers everything, because the basis itself grows in
  /// every position rather than only at the end.
  pub fn rank(&self) -> usize {
    self.rank
  }

  /// Inverse to [`Self::rank`].
  pub fn from_rank(radix: usize, degree: usize, rank: usize) -> Self {
    Self {
      rank,
      degree,
      radix,
    }
  }

  /// Every word of a degree over an alphabet, in rank order.
  ///
  /// Total at the trivial ends: degree zero gives the one empty word, and a
  /// positive degree over the empty alphabet gives nothing.
  pub fn all(radix: usize, degree: usize) -> Words {
    Words {
      radix,
      degree,
      next: 0,
      count: Self::count(radix, degree),
    }
  }

  /// Concatenation: the product of the tensor algebra.
  ///
  /// Total and unsigned, where the monotone families are partial (a repeat
  /// annihilates an alternating index) or signed. Nothing is reordered, so
  /// there is no permutation to take a sign from.
  ///
  /// # Panics
  /// If the alphabets differ.
  pub fn concat(&self, other: &Self) -> Self {
    assert_eq!(
      self.radix, other.radix,
      "a concatenation is over one alphabet"
    );
    Self {
      rank: self.rank * self.radix.pow(other.degree as u32) + other.rank,
      degree: self.degree + other.degree,
      radix: self.radix,
    }
  }

  /// Every single-position deletion, by position.
  ///
  /// The free counterpart of [`MonoIndex::deletions`](crate::MonoIndex::deletions),
  /// and the simplest of the three: positions are distinct and carry no sign,
  /// so each yields one reduced word. Contraction of a free slot is this.
  pub fn deletions(&self) -> WordDeletions {
    WordDeletions {
      index: *self,
      position: 0,
    }
  }
}

/// Every [`Word`] of a degree and alphabet, in radix order.
///
/// A named type rather than an opaque one so a caller dispatching between the
/// families can hold either without boxing.
#[derive(Debug, Clone)]
pub struct Words {
  radix: usize,
  degree: usize,
  next: usize,
  count: usize,
}

impl Iterator for Words {
  type Item = Word;
  fn next(&mut self) -> Option<Word> {
    if self.next == self.count {
      return None;
    }
    let word = Word::from_rank(self.radix, self.degree, self.next);
    self.next += 1;
    Some(word)
  }
  fn size_hint(&self) -> (usize, Option<usize>) {
    let remaining = self.count - self.next;
    (remaining, Some(remaining))
  }
}
impl ExactSizeIterator for Words {}

/// Every single-position deletion of a [`Word`].
///
/// Owns the index, which is [`Copy`] now that it is a packed rank.
#[derive(Debug, Clone)]
pub struct WordDeletions {
  index: Word,
  position: usize,
}

impl Iterator for WordDeletions {
  type Item = (usize, Word);
  fn next(&mut self) -> Option<Self::Item> {
    if self.position == self.index.degree {
      return None;
    }
    let position = self.position;
    self.position += 1;

    // Split the rank at the position: the digits above it keep their place
    // once the removed one is taken out, and those below shift up.
    let below_width = self.index.degree - 1 - position;
    let scale = self.index.radix.pow(below_width as u32);
    let above = self.index.rank / (scale * self.index.radix);
    let symbol = (self.index.rank / scale) % self.index.radix;
    let below = self.index.rank % scale;
    Some((
      symbol,
      Word {
        rank: above * scale + below,
        degree: self.index.degree - 1,
        radix: self.index.radix,
      },
    ))
  }
}

#[cfg(test)]
mod test {
  use super::*;

  /// Ranking is a bijection onto `0..n^k`, and enumeration walks it in order.
  #[test]
  fn ranking_is_the_radix_bijection() {
    for radix in 0..=3 {
      for degree in 0..=3 {
        let words: Vec<Word> = Word::all(radix, degree).collect();
        assert_eq!(words.len(), Word::count(radix, degree));
        for (rank, word) in words.iter().enumerate() {
          assert_eq!(word.rank(), rank);
          assert_eq!(word.degree(), degree);
        }
      }
    }
  }

  /// The last position runs fastest, matching the stride convention every
  /// graded structure in the workspace uses.
  #[test]
  fn the_last_position_runs_fastest() {
    let radix = 3;
    let first = Word::new(radix, [1, 0]);
    let last = Word::new(radix, [0, 1]);
    assert_eq!(first.rank(), radix);
    assert_eq!(last.rank(), 1);
  }

  /// Concatenation is associative with the empty word as its unit, and unlike
  /// the monotone merges it is total: no word annihilates another.
  #[test]
  fn concatenation_is_a_total_monoid() {
    let radix = 3;
    let unit = Word::empty(radix);
    for a in Word::all(radix, 2) {
      assert_eq!(a.concat(&unit), a);
      assert_eq!(unit.concat(&a), a);
      for b in Word::all(radix, 2) {
        for c in Word::all(radix, 1) {
          assert_eq!(a.concat(&b).concat(&c), a.concat(&b.concat(&c)));
        }
        // Order matters, where a symmetric merge would identify the two.
        if a != b {
          assert_ne!(a.concat(&b), b.concat(&a));
        }
      }
    }
  }

  /// A word of degree k has exactly k deletions, one per position, and a
  /// repeated symbol yields the same reduced word more than once, which is the
  /// multiplicity a contraction must count.
  #[test]
  fn deletions_are_positional() {
    let word = Word::new(3, [2, 0, 2]);
    let deletions: Vec<(usize, Word)> = word.deletions().collect();
    assert_eq!(deletions.len(), 3);
    assert_eq!(deletions[0], (2, Word::new(3, [0, 2])));
    assert_eq!(deletions[1], (0, Word::new(3, [2, 2])));
    assert_eq!(deletions[2], (2, Word::new(3, [2, 0])));

    assert_eq!(Word::empty(3).deletions().count(), 0);
  }
}

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

/// The whole grid ${0, dots, "radix"-1}^"dim"$, in linear-index order.
///
/// Colexicographic on the digits, the crate-wide convention: the least
/// significant axis varies fastest. The $i$-th item is
/// [`linear2cartesian`]`(i, radix, dim)`.
///
/// Total at the degenerate ends: dimension $0$ yields the one empty index (the
/// single point of a $0$-fold product), radix $0$ in positive dimension yields
/// nothing.
pub fn grid(radix: usize, dim: usize) -> impl Iterator<Item = Vec<usize>> {
  (0..radix.pow(dim as u32)).map(move |i| linear2cartesian(i, radix, dim))
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

/// Converts a cartesian multi-index to a linear index when the axes carry
/// different radices: the positional number of mixed base
/// $"radix"_0, dots, "radix"_(d-1)$, least significant axis first.
///
/// The uniform [`cartesian2linear`] is the constant-radix case.
pub fn cartesian2linear_mixed(cart_idx: &[usize], radices: &[usize]) -> usize {
  let mut lin_idx = 0;
  for (&icomp, &radix) in cart_idx.iter().zip(radices).rev() {
    lin_idx *= radix;
    lin_idx += icomp;
  }
  lin_idx
}

/// The inverse of [`cartesian2linear_mixed`]: the mixed-radix digits of a
/// linear index in `0..radices.product()`.
pub fn linear2cartesian_mixed(mut lin_idx: usize, radices: &[usize]) -> Vec<usize> {
  radices
    .iter()
    .map(|&radix| {
      let digit = lin_idx % radix;
      lin_idx /= radix;
      digit
    })
    .collect()
}

/// The linear-index offset of a cube corner (a set of axes with coordinate 1)
/// under the given per-axis strides.
pub fn corner_offset(corner: Combination, strides: &[usize]) -> usize {
  corner.iter().map(|axis| strides[axis]).sum()
}

/// The per-axis strides of a mixed-radix linear index: the running product
/// $"stride"_i = product_(j < i) "radix"_j$.
///
/// The uniform [`strides`] is the constant-radix case, $"radix"^i$.
pub fn mixed_strides(radices: &[usize]) -> Vec<usize> {
  radices
    .iter()
    .scan(1, |stride, &radix| {
      let this = *stride;
      *stride *= radix;
      Some(this)
    })
    .collect()
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

  /// The grid enumerates every multi-index exactly once, in linear-index
  /// order, and is total at the degenerate ends.
  #[test]
  fn grid_enumerates_the_product() {
    for radix in 0usize..=4 {
      for dim in 0..=4 {
        let all: Vec<Vec<usize>> = grid(radix, dim).collect();
        assert_eq!(all.len(), radix.pow(dim as u32), "radix {radix}, dim {dim}");
        for (i, cart) in all.iter().enumerate() {
          assert_eq!(cart.len(), dim);
          assert_eq!(cartesian2linear(cart, radix), i);
        }
        let mut distinct = all.clone();
        distinct.sort();
        distinct.dedup();
        assert_eq!(distinct.len(), all.len());
      }
    }
    assert_eq!(grid(0, 0).collect::<Vec<_>>(), vec![Vec::<usize>::new()]);
  }

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

  /// Mixed-radix linearization is a bijection onto `0..radices.product()`, and
  /// reduces to the uniform one when every radix agrees.
  #[test]
  fn mixed_radix_round_trip() {
    for radices in [
      vec![],
      vec![1],
      vec![4],
      vec![2, 3],
      vec![3, 1, 4],
      vec![2, 2, 2],
    ] {
      let count: usize = radices.iter().product();
      for lin in 0..count {
        let cart = linear2cartesian_mixed(lin, &radices);
        assert!(cart.iter().zip(&radices).all(|(&c, &r)| c < r));
        assert_eq!(cartesian2linear_mixed(&cart, &radices), lin);
      }
      if let Ok(&radix) = radices.iter().all_equal_value() {
        for lin in 0..count {
          assert_eq!(
            linear2cartesian_mixed(lin, &radices),
            linear2cartesian(lin, radix, radices.len())
          );
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
