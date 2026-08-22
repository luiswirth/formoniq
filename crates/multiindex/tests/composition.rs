//! Weak compositions, the basis of $"Sym"^d$, colex-ranked.

use multiindex::{Composition, binomial};

/// Stars and bars: the weak compositions of $d$ into $p$ parts number
/// $binom(p + d - 1, d) = dim "Sym"^d (RR^p)$, and the enumeration realizes
/// that count with no duplicates and the declared shape throughout.
#[test]
fn count_is_the_symmetric_power_dimension() {
  for nparts in 0..=5 {
    for degree in 0..=6 {
      let all: Vec<_> = Composition::all(nparts, degree).collect();
      assert_eq!(all.len(), Composition::count(nparts, degree));
      // The empty alphabet admits only the empty composition, which is where
      // the stars-and-bars formula has no bars to place.
      let expected = if nparts == 0 {
        usize::from(degree == 0)
      } else {
        binomial(nparts + degree - 1, degree)
      };
      assert_eq!(all.len(), expected);
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
