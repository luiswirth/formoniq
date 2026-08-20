//! Weak compositions, the basis of $"Sym"^d$, colex-ranked.

use multiindex::{Composition, combinations};

/// The enumeration has the dimension of $"Sym"^d (RR^p)$, is duplicate-free,
/// and every element has the declared shape.
#[test]
fn count_is_the_symmetric_power_dimension() {
  for nparts in 0..=5 {
    for degree in 0..=6 {
      let all: Vec<_> = Composition::all(nparts, degree).collect();
      assert_eq!(all.len(), Composition::count(nparts, degree));
      for composition in &all {
        assert_eq!(composition.nparts(), nparts);
        assert_eq!(composition.degree(), degree);
      }
      let mut unique = all.clone();
      unique.sort();
      unique.dedup();
      assert_eq!(unique.len(), all.len());
    }
  }
}

/// Ranking is the position in the enumeration, and inverts it.
#[test]
fn rank_inverts_the_enumeration() {
  for nparts in 0..=5 {
    for degree in 0..=6 {
      for (i, composition) in Composition::all(nparts, degree).enumerate() {
        assert_eq!(composition.rank(), i);
        assert_eq!(Composition::from_rank(nparts, degree, i), composition);
      }
    }
  }
}

/// The enumeration is colexicographic on the word: compare the largest
/// symbol first, and the smallest last.
#[test]
fn order_is_colexicographic_on_the_word() {
  let colex_key = |composition: &Composition| {
    let mut word = composition.word();
    word.reverse();
    word
  };
  for nparts in 0..=5 {
    for degree in 0..=6 {
      let all: Vec<_> = Composition::all(nparts, degree).collect();
      for pair in all.windows(2) {
        assert!(colex_key(&pair[0]) < colex_key(&pair[1]));
      }
    }
  }
}

/// A rank is independent of the number of parts: adding parts leaves every
/// existing composition where it was.
///
/// This is what colex is for. Under a reverse-lexicographic order a word's
/// position drifts upward as parts are appended, so widening the alphabet
/// renumbers the basis. The formula makes it plain: the sum runs over the word
/// and never mentions `nparts`.
#[test]
fn rank_does_not_depend_on_the_number_of_parts() {
  for degree in 0..=4 {
    for nparts in 1..=4 {
      for composition in Composition::all(nparts, degree) {
        for wider in nparts..=6 {
          let widened = Composition::from_word(wider, &composition.word());
          assert_eq!(widened.rank(), composition.rank());
        }
      }
    }
  }
}

/// Stars and bars: compositions of degree $d$ into $p$ parts biject with the
/// $d$-subsets of $d + p - 1$, order for order, by the shift
/// $w_i |-> w_i + i$ on the word.
///
/// The stars are the subset here, not the bars. Both readings biject, and
/// they are complementary, but only this one is order-preserving under the
/// shared colex convention: a bar set has $p - 1$ elements, so its rank
/// depends on the number of parts, while the word has $d$ and its rank does
/// not.
///
/// A theorem about the two index sets, not the way either is built, which
/// is what leaves the degree unbounded here while a combination's index
/// count stays bounded.
#[test]
fn stars_and_bars_bijects_with_combinations() {
  for nparts in 1..=5 {
    for degree in 0..=6 {
      let slots = degree + nparts - 1;
      let via_stars: Vec<Composition> = combinations(slots, degree)
        .map(|star_set| {
          let word: Vec<usize> = star_set
            .iter()
            .enumerate()
            .map(|(position, symbol)| symbol - position)
            .collect();
          Composition::from_word(nparts, &word)
        })
        .collect();
      assert_eq!(
        via_stars,
        Composition::all(nparts, degree).collect::<Vec<_>>()
      );
    }
  }
}

/// The graded monoid: addition is associative, the zero composition is its
/// unit, and degrees add.
#[test]
fn addition_is_a_graded_monoid() {
  for nparts in 0..=4 {
    let zero = Composition::zero(nparts);
    for a in Composition::all(nparts, 3) {
      assert_eq!(&a + &zero, a);
      assert_eq!(&zero + &a, a);
      for b in Composition::all(nparts, 2) {
        let sum = &a + &b;
        assert_eq!(sum.degree(), a.degree() + b.degree());
        for c in Composition::all(nparts, 1) {
          assert_eq!(&(&a + &b) + &c, &a + &(&b + &c));
        }
      }
    }
  }
}

/// The degree is genuinely unbounded: past the bitset ceiling a
/// [`MonoIndex`](multiindex::MonoIndex) imposes on the shifted alphabet, which
/// is exactly the bound stars and bars would have inherited.
#[test]
fn degree_is_unbounded() {
  for degree in [63, 64, 65, 256] {
    let all: Vec<_> = Composition::all(2, degree).collect();
    assert_eq!(all.len(), degree + 1);
    assert_eq!(all[0].parts(), &[degree, 0]);
    assert_eq!(all[degree].parts(), &[0, degree]);
  }
  assert_eq!(Composition::all(4, 100).count(), Composition::count(4, 100));
}
