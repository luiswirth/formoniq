//! Cartesian multi-indices: positional (radix) numbers.
//!
//! Elements of a product $product_i {0, dots, r_i - 1}$, the index sets of
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
//!
//! [`Radix`] is the shape of such an index set and owns the arithmetic; a
//! [`Word`] is one index of a shape whose axes all agree, and delegates. The
//! order is colexicographic on the digits, as everywhere in this crate: axis
//! $0$ runs fastest and $s_i = product_(j < i) r_j$.

use super::Combination;
use crate::monotone::Symbols;

/// The shape of a cartesian index set: the radix of each axis.
///
/// The index set is $product_i {0, dots, r_i - 1}$, whose elements are the
/// digit tuples and whose cardinality is [`Self::count`]. A grid of cells, the
/// vertices of a box, the component index of a tensor product of spaces of
/// differing dimension: each is a shape, held once, against which individual
/// indices are linearized and delinearized.
///
/// The shape belongs to the index *set*, not to an index, which is why it lives
/// here rather than inside every digit tuple. [`Word`] is the exception, an
/// index of a uniform shape that carries enough to rank itself.
///
/// Total at the degenerate corners: no axes give the one empty index (the single
/// point of an empty product), and an axis of radix zero gives none.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Radix {
  radices: Symbols,
}

impl Radix {
  pub fn new(radices: impl IntoIterator<Item = usize>) -> Self {
    Self {
      radices: radices.into_iter().collect(),
    }
  }
  /// The shape whose `naxes` axes all have the same radix: the index set
  /// ${0, dots, "radix" - 1}^"naxes"$, hence the basis of a tensor power.
  pub fn uniform(radix: usize, naxes: usize) -> Self {
    Self::new(std::iter::repeat_n(radix, naxes))
  }

  pub fn naxes(&self) -> usize {
    self.radices.len()
  }
  pub fn radices(&self) -> &[usize] {
    &self.radices
  }
  /// The radix common to every axis, if there is one.
  pub fn uniform_radix(&self) -> Option<usize> {
    let first = *self.radices.first()?;
    self.radices.iter().all(|&r| r == first).then_some(first)
  }

  /// The number of indices, $product_i r_i$: the dimension of the product.
  pub fn count(&self) -> usize {
    self.radices.iter().product()
  }

  /// The stride of each axis in the linear index, the running product
  /// $s_i = product_(j < i) r_j$.
  ///
  /// Axis $0$ has stride $1$ and so runs fastest, which is what makes the
  /// linear order colexicographic on the digits.
  pub fn strides(&self) -> Symbols {
    self
      .radices
      .iter()
      .scan(1, |stride, &radix| {
        let this = *stride;
        *stride *= radix;
        Some(this)
      })
      .collect()
  }

  /// The linear index of a digit tuple, $sum_i d_i s_i$.
  ///
  /// # Panics
  /// If the tuple has the wrong number of axes.
  pub fn linearize(&self, digits: &[usize]) -> usize {
    assert_eq!(
      digits.len(),
      self.naxes(),
      "digit tuple has the wrong shape"
    );
    let mut linear = 0;
    for (&digit, &radix) in digits.iter().zip(&self.radices).rev() {
      linear = linear * radix + digit;
    }
    linear
  }

  /// The digit tuple of a linear index. Inverse to [`Self::linearize`].
  pub fn delinearize(&self, mut linear: usize) -> Symbols {
    self
      .radices
      .iter()
      .map(|&radix| {
        let digit = linear % radix;
        linear /= radix;
        digit
      })
      .collect()
  }

  /// Every index of the shape, in linear order: the $i$-th item is
  /// [`Self::delinearize`]`(i)`.
  pub fn all(&self) -> impl Iterator<Item = Symbols> + Clone + use<> {
    let shape = self.clone();
    (0..shape.count()).map(move |linear| shape.delinearize(linear))
  }

  /// The linear index of a cube corner, a set of axes with digit $1$ and the
  /// rest $0$: $sum_(i in "corner") s_i$.
  ///
  /// The radix-2 reading of a [`Combination`], and the offset a Kuhn cell's
  /// vertices are found at within a grid.
  pub fn corner_offset(&self, corner: Combination) -> usize {
    let strides = self.strides();
    corner.iter().map(|axis| strides[axis]).sum()
  }
}

impl FromIterator<usize> for Radix {
  fn from_iter<T: IntoIterator<Item = usize>>(iter: T) -> Self {
    Self::new(iter)
  }
}

/// A word over one alphabet: a basis element of $V^(times.circle k)$.
///
/// Carries its own alphabet, unlike [`MonoIndex`](crate::MonoIndex), because a
/// radix rank cannot be taken without one. That asymmetry is forced: the colex
/// rank of a monotone index is a sum of binomials in its symbols alone, while
/// the rank of a word is a positional number and positions have a base.
///
/// One radix rather than a [`Radix`] of its own, so it stays [`Copy`] and
/// allocation-free: a word indexes a power of a single space, where the axes
/// agree by construction. A product of *differing* spaces is a shape, and the
/// shape is held once by whoever owns the index set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Word {
  /// The word packed as its own radix rank.
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
    let symbols: Symbols = symbols.into_iter().collect();
    assert!(
      symbols.iter().all(|&symbol| symbol < radix),
      "a word's symbols lie below its radix"
    );
    Self {
      rank: Radix::uniform(radix, symbols.len()).linearize(&symbols),
      degree: symbols.len(),
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
  /// The shape this word is an index of.
  pub fn shape(&self) -> Radix {
    Radix::uniform(self.radix, self.degree)
  }
  /// The symbols, in position order, unpacked from the rank.
  pub fn symbols(&self) -> Symbols {
    self.shape().delinearize(self.rank)
  }
  pub fn symbol(&self, position: usize) -> usize {
    assert!(position < self.degree, "position out of range");
    (self.rank / self.radix.pow(position as u32)) % self.radix
  }
  pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
    (0..self.degree).map(move |position| self.symbol(position))
  }
  /// How many positions carry a symbol.
  pub fn multiplicity(&self, symbol: usize) -> usize {
    self.iter().filter(|&s| s == symbol).count()
  }

  /// The number of words of a degree over an alphabet: $n^k$.
  ///
  /// The dimension of $V^(times.circle k)$, and the count no symmetry reduces.
  pub fn count(radix: usize, degree: usize) -> usize {
    radix.pow(degree as u32)
  }

  /// The radix rank, position $0$ running fastest: $sum_j w_j space n^j$.
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
      rank: self.rank + other.rank * self.radix.pow(self.degree as u32),
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

    // Split the rank at the position: the digits below it keep their place once
    // the removed one is taken out, and those above slide down one stride.
    let stride = self.index.radix.pow(position as u32);
    let below = self.index.rank % stride;
    let symbol = (self.index.rank / stride) % self.index.radix;
    let above = self.index.rank / (stride * self.index.radix);
    Some((
      symbol,
      Word {
        rank: below + above * stride,
        degree: self.index.degree - 1,
        radix: self.index.radix,
      },
    ))
  }
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::combinations;
  use itertools::Itertools;

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
          assert_eq!(Word::new(radix, word.symbols()), *word);
        }
      }
    }
  }

  /// Position zero runs fastest, the colex convention of the whole crate.
  #[test]
  fn the_first_position_runs_fastest() {
    let radix = 3;
    assert_eq!(Word::new(radix, [1, 0]).rank(), 1);
    assert_eq!(Word::new(radix, [0, 1]).rank(), radix);
  }

  /// A word is one index of a uniform shape, and the two agree on the
  /// arithmetic: the word ranks itself exactly as its shape linearizes it.
  #[test]
  fn a_word_is_an_index_of_its_shape() {
    for radix in 1..=3 {
      for degree in 0..=3 {
        let shape = Radix::uniform(radix, degree);
        assert_eq!(shape.count(), Word::count(radix, degree));
        for (linear, digits) in shape.all().enumerate() {
          let word = Word::new(radix, digits.iter().copied());
          assert_eq!(word.rank(), linear);
          assert_eq!(word.symbols(), digits);
          assert_eq!(shape.linearize(&digits), linear);
        }
      }
    }
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
      // Concatenation is the juxtaposition of the symbols, in order.
      for b in Word::all(radix, 1) {
        let joined: Symbols = a.iter().chain(b.iter()).collect();
        assert_eq!(a.concat(&b), Word::new(radix, joined));
      }
    }
  }

  /// A word of degree k has exactly k deletions, one per position, and a
  /// repeated symbol yields the same reduced word more than once, which is the
  /// multiplicity a contraction must count.
  #[test]
  fn deletions_are_positional() {
    for radix in 1..=3 {
      for degree in 0..=3 {
        for word in Word::all(radix, degree) {
          let deletions: Vec<(usize, Word)> = word.deletions().collect();
          assert_eq!(deletions.len(), degree);
          for (position, &(symbol, reduced)) in deletions.iter().enumerate() {
            assert_eq!(symbol, word.symbol(position));
            let expected: Symbols = word
              .iter()
              .enumerate()
              .filter_map(|(i, s)| (i != position).then_some(s))
              .collect();
            assert_eq!(reduced, Word::new(radix, expected));
          }
        }
      }
    }
    assert_eq!(Word::empty(3).deletions().count(), 0);
  }

  /// Linearization is a bijection onto `0..count`, every digit stays in range,
  /// and the enumeration walks it in order. Includes the degenerate shapes:
  /// no axes give the one empty index, an axis of radix zero give none.
  #[test]
  fn linearization_is_the_mixed_radix_bijection() {
    for radices in [
      vec![],
      vec![0],
      vec![1],
      vec![4],
      vec![2, 3],
      vec![3, 1, 4],
      vec![2, 2, 2],
    ] {
      let shape = Radix::new(radices.iter().copied());
      assert_eq!(shape.count(), radices.iter().product::<usize>());
      let all: Vec<Symbols> = shape.all().collect();
      assert_eq!(all.len(), shape.count());
      for (linear, digits) in all.iter().enumerate() {
        assert!(digits.iter().zip(&radices).all(|(&d, &r)| d < r));
        assert_eq!(shape.linearize(digits), linear);
        assert_eq!(shape.delinearize(linear), *digits);
      }
      let mut distinct = all.clone();
      distinct.sort();
      distinct.dedup();
      assert_eq!(distinct.len(), all.len());
    }
    assert_eq!(
      Radix::new([]).all().collect::<Vec<_>>(),
      vec![Symbols::new()]
    );
  }

  /// The strides are the running product, reconstruct the linear index as
  /// $sum_i d_i s_i$, and reduce to the radix powers on a uniform shape.
  #[test]
  fn strides_are_the_running_product() {
    for radices in [vec![], vec![4], vec![2, 3], vec![3, 1, 4], vec![2, 2, 2]] {
      let shape = Radix::new(radices.iter().copied());
      let strides = shape.strides();
      assert_eq!(strides.len(), shape.naxes());
      for (axis, &stride) in strides.iter().enumerate() {
        assert_eq!(stride, radices[..axis].iter().product::<usize>());
      }
      for (linear, digits) in shape.all().enumerate() {
        let weighted: usize = digits.iter().zip(&strides).map(|(&d, &s)| d * s).sum();
        assert_eq!(weighted, linear);
      }
      if let Ok(&radix) = radices.iter().all_equal_value() {
        let uniform = Radix::uniform(radix, radices.len());
        assert_eq!(uniform, shape);
        assert_eq!(shape.uniform_radix(), Some(radix));
        for (axis, &stride) in strides.iter().enumerate() {
          assert_eq!(stride, radix.pow(axis as u32));
        }
      }
    }
  }

  /// A cube corner is a radix-2 cartesian index: its stride offset equals the
  /// linear index of the 0/1 indicator vector of the chosen axes.
  #[test]
  fn corner_offset_is_the_indicator_linear_index() {
    for dim in 0..=4 {
      let shape = Radix::uniform(2, dim);
      for card in 0..=dim {
        for corner in combinations(dim, card) {
          let indicator: Symbols = (0..dim)
            .map(|axis| usize::from(corner.contains(axis)))
            .collect();
          assert_eq!(shape.corner_offset(corner), shape.linearize(&indicator));
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
