//! Weak compositions: the multi-indices of the symmetric algebra.
//!
//! A [`Composition`] is a tuple $k in NN_0^p$ with $sum_i k_i = d$: the
//! exponent vector of the monomial $x^k$, hence the canonical basis of
//! $"Sym"^d (RR^p)$, the degree-$d$ part of the polynomial algebra.
//!
//! This is the symmetric counterpart of [`Combination`](crate::Combination),
//! which indexes $Lambda^k$. The two are dual in structure, not variants of one
//! thing: a combination forbids repetition and carries a
//! [`Sign`](crate::Sign) under permutation, a composition mandates neither and
//! carries no sign. Compositions form a graded monoid under addition --
//! $x^k x^(k') = x^(k + k')$ -- where combinations instead carry the wedge,
//! which is partial and signed.
//!
//! Stars and bars is a bijection onto the $(p-1)$-subsets of $d + p - 1$ slots,
//! and it is proved as a theorem here rather than used as the representation.
//! It is not natural in the ambient size: it absorbs the *degree* $d$, which is
//! unbounded (a refinement level, a polynomial order), into the *index count* of
//! a combination, which is bounded by a dimension. Enumerating compositions
//! directly is what keeps the degree free.

use crate::binomial;

/// A weak composition $k in NN_0^p$ of degree $d = sum_i k_i$: the exponent
/// vector of the monomial $x^k$, a basis element of $"Sym"^d (RR^p)$.
///
/// The degree is unbounded. Order among compositions of a fixed shape is
/// reverse-lexicographic on the parts (see [`Composition::all`]), which is the
/// colex order of the corresponding bar sets and hence the crate's one
/// indexing convention.
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct Composition {
  /// The parts. Their sum is the degree; the length is the number of parts.
  parts: Vec<usize>,
}

impl Composition {
  pub fn new(parts: Vec<usize>) -> Self {
    Self { parts }
  }
  /// The zero composition of $p$ parts: the unit of the monoid, the monomial
  /// $1$.
  pub fn zero(nparts: usize) -> Self {
    Self::new(vec![0; nparts])
  }

  pub fn nparts(&self) -> usize {
    self.parts.len()
  }
  /// The degree $d = sum_i k_i$: the total degree of the monomial $x^k$.
  pub fn degree(&self) -> usize {
    self.parts.iter().sum()
  }
  pub fn parts(&self) -> &[usize] {
    &self.parts
  }
  pub fn into_parts(self) -> Vec<usize> {
    self.parts
  }

  /// The number of compositions of degree `degree` into `nparts` parts,
  /// $binom(d + p - 1, p - 1)$ -- equivalently $dim "Sym"^d (RR^p)$.
  ///
  /// Total at the degenerate corners: no parts admit only the empty
  /// composition of degree zero.
  pub fn count(nparts: usize, degree: usize) -> usize {
    if nparts == 0 {
      usize::from(degree == 0)
    } else {
      binomial(degree + nparts - 1, nparts - 1)
    }
  }

  /// Every composition of degree `degree` into `nparts` parts, in
  /// reverse-lexicographic order on the parts: the leading part descends from
  /// `degree` to $0$, each block ordered the same way in the parts that remain.
  ///
  /// This is the colex order of the bar sets under stars and bars, so it agrees
  /// with [`Combination`](crate::Combination)'s convention; at degree one it
  /// lists $e_0, e_1, dots$, the order the standard basis places them in.
  /// Enumerated by successor on the parts, so nothing bounds the degree.
  pub fn all(nparts: usize, degree: usize) -> impl Iterator<Item = Composition> {
    Compositions {
      current: (nparts > 0 || degree == 0).then(|| {
        let mut parts = vec![0; nparts];
        if nparts > 0 {
          parts[0] = degree;
        }
        Composition::new(parts)
      }),
      degree,
    }
  }

  /// The position of this composition in [`Composition::all`], its canonical
  /// index. Inverse to [`Composition::from_rank`].
  pub fn rank(&self) -> usize {
    let mut rank = 0;
    let mut remaining = self.degree();
    for (i, &part) in self.parts.iter().enumerate() {
      let rest = self.nparts() - i - 1;
      // The blocks skipped by taking this part rather than a larger one: every
      // leading part above `part` heads a full block of the shorter shape.
      for larger in (part + 1)..=remaining {
        rank += Self::count(rest, remaining - larger);
      }
      remaining -= part;
    }
    rank
  }

  /// The composition of degree `degree` into `nparts` parts at position `rank`
  /// of [`Composition::all`]. Inverse to [`Composition::rank`].
  ///
  /// # Panics
  /// If `rank` is not below [`Composition::count`].
  pub fn from_rank(nparts: usize, degree: usize, rank: usize) -> Self {
    assert!(rank < Self::count(nparts, degree), "rank out of range");
    let mut rank = rank;
    let mut remaining = degree;
    let mut parts = Vec::with_capacity(nparts);
    for i in 0..nparts {
      let rest = nparts - i - 1;
      let mut part = remaining;
      loop {
        let block = Self::count(rest, remaining - part);
        if rank < block {
          break;
        }
        rank -= block;
        part -= 1;
      }
      parts.push(part);
      remaining -= part;
    }
    Self::new(parts)
  }
}

/// Successor enumeration of [`Composition::all`].
struct Compositions {
  current: Option<Composition>,
  degree: usize,
}

impl Iterator for Compositions {
  type Item = Composition;
  fn next(&mut self) -> Option<Composition> {
    let current = self.current.take()?;
    // The successor moves one unit from the last nonzero part that is not the
    // final one, and sweeps everything past it into the part just after --
    // reverse-lexicographic descent. The final part holding the whole degree is
    // the last composition.
    let parts = current.parts();
    let last = parts.len().checked_sub(1);
    self.current = last
      .filter(|&last| parts[last] != self.degree)
      .and_then(|last| {
        let pivot = parts[..last].iter().rposition(|&part| part > 0)?;
        let mut next = parts.to_vec();
        let carry = next[last] + 1;
        next[last] = 0;
        next[pivot] -= 1;
        next[pivot + 1] += carry;
        Some(Composition::new(next))
      });
    Some(current)
  }
}

impl std::ops::Add for &Composition {
  type Output = Composition;
  /// Monomial multiplication $x^k x^(k') = x^(k + k')$: the graded monoid, of
  /// degree the sum of the degrees.
  ///
  /// # Panics
  /// If the shapes differ -- the two must be compositions into the same parts.
  fn add(self, other: &Composition) -> Composition {
    assert_eq!(
      self.nparts(),
      other.nparts(),
      "compositions add within a fixed number of parts"
    );
    Composition::new(
      self
        .parts
        .iter()
        .zip(&other.parts)
        .map(|(a, b)| a + b)
        .collect(),
    )
  }
}

impl FromIterator<usize> for Composition {
  fn from_iter<T: IntoIterator<Item = usize>>(iter: T) -> Self {
    Self::new(iter.into_iter().collect())
  }
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::combinations;

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

  /// The enumeration descends reverse-lexicographically on the parts.
  #[test]
  fn order_is_reverse_lexicographic() {
    for nparts in 0..=5 {
      for degree in 0..=6 {
        let all: Vec<_> = Composition::all(nparts, degree).collect();
        for pair in all.windows(2) {
          assert!(pair[0].parts() > pair[1].parts());
        }
      }
    }
  }

  /// Stars and bars: compositions of degree $d$ into $p$ parts biject with the
  /// $(p-1)$-subsets of $d + p - 1$, order for order. A theorem about the two
  /// index sets, not the way either is built -- which is what leaves the degree
  /// unbounded here while a combination's index count stays bounded.
  #[test]
  fn stars_and_bars_bijects_with_combinations() {
    for nparts in 1..=5 {
      for degree in 0..=6 {
        let bars = nparts - 1;
        let slots = degree + bars;
        let via_bars: Vec<Composition> = combinations(slots, bars)
          .take(binomial(slots, bars))
          .map(|bar_set| {
            // The gaps the bars cut the slots into, reversed: the leading part
            // is the last gap.
            let mut parts = Vec::with_capacity(nparts);
            let mut previous = None;
            for bar in bar_set.iter() {
              parts.push(bar - previous.map_or(0, |p| p + 1));
              previous = Some(bar);
            }
            parts.push(slots - previous.map_or(0, |p| p + 1));
            parts.reverse();
            Composition::new(parts)
          })
          .collect();
        assert_eq!(
          via_bars,
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

  /// The degree is genuinely unbounded: past the 64-index ceiling a
  /// [`Combination`](crate::Combination) imposes, which is exactly the bound
  /// stars and bars would have inherited.
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
}
