//! Radix multi-indices, and the Kuhn triangulation they represent.

use itertools::Itertools;
use multiindex::{Combination, Radix, Symbols, Word, combinations};

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
