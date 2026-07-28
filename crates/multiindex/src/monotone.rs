//! The two families of multi-indices as one object.
//!
//! A [`Combination`](crate::Combination) is a strictly increasing word and a
//! [`Composition`](crate::Composition) a weakly increasing one, over the same
//! alphabet and of the same length. That is the entire difference between
//! them, and [`Repetition`] is the bit carrying it: forbidden gives the
//! subsets, allowed the multisets.
//!
//! Counting, ranking and enumeration are then *one* implementation with a
//! position-dependent offset, not two behind a shared signature. The offset is
//! the classical shift $w_i |-> w_i + i$ taking a weakly increasing word to a
//! strictly increasing one, and it appears here only inside the arithmetic --
//! never as a stored representation, which is the distinction the workspace
//! keeps between a theorem and a data structure.

use crate::binomial;

/// Whether a multi-index may repeat a symbol.
///
/// Forbidden gives the strictly increasing words, the $binom(n, k)$ subsets;
/// allowed gives the weakly increasing ones, the $binom(n + k - 1, k)$
/// multisets. The two are otherwise the same object, and every operation here
/// is written once over both.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Repetition {
  Forbidden,
  Allowed,
}

impl Repetition {
  /// The offset separating the two families: the amount by which the symbol at
  /// `position` is raised to make a weakly increasing word strictly
  /// increasing.
  ///
  /// Zero when repetition is forbidden, which is why one implementation covers
  /// both. Every other function in this module is the strictly increasing case
  /// composed with this.
  pub fn shift(self, position: usize) -> usize {
    match self {
      Repetition::Forbidden => 0,
      Repetition::Allowed => position,
    }
  }

  /// The alphabet the shifted word ranges over: `nsymbols` unchanged when
  /// repetition is forbidden, widened by the shift when it is allowed.
  fn shifted_nsymbols(self, nsymbols: usize, degree: usize) -> usize {
    nsymbols + self.shift(degree.saturating_sub(1))
  }

  /// The number of monotone words of length `degree` over `nsymbols` symbols:
  /// $binom(n, k)$ forbidden and $binom(n + k - 1, k)$ allowed.
  ///
  /// One binomial, the family entering only through [`Self::shift`]. Total at
  /// the degenerate corners: the empty word is the unique word of degree zero
  /// over any alphabet, and no alphabet admits no others.
  pub fn count(self, nsymbols: usize, degree: usize) -> usize {
    binomial(self.shifted_nsymbols(nsymbols, degree), degree)
  }

  /// The colexicographic rank of a monotone word, its canonical index.
  ///
  /// The combinatorial number system $sum_i binom(w_i + "shift"(i), i + 1)$,
  /// applied to the shifted word. Independent of the alphabet size, so
  /// widening the alphabet renumbers nothing already there -- the property
  /// that makes colex the workspace's convention.
  ///
  /// # Panics
  /// If the word is not monotone for this family.
  pub fn rank(self, word: &[usize]) -> usize {
    assert!(self.is_monotone(word), "word is not monotone for {self:?}");
    word
      .iter()
      .enumerate()
      .map(|(position, &symbol)| binomial(symbol + self.shift(position), position + 1))
      .sum()
  }

  /// The monotone word at the given colexicographic rank. Inverse to
  /// [`Self::rank`].
  ///
  /// The combinatorial number system read backwards: the highest position
  /// takes the largest symbol its binomial still fits under, and the remainder
  /// passes down. Greedy and exact, so it costs the degree rather than the
  /// rank.
  ///
  /// # Panics
  /// If the rank is not below [`Self::count`].
  pub fn word_from_rank(self, nsymbols: usize, degree: usize, rank: usize) -> Vec<usize> {
    assert!(
      rank < self.count(nsymbols, degree),
      "rank out of range for {self:?}"
    );
    let mut remaining = rank;
    let mut shifted = vec![0; degree];
    for position in (0..degree).rev() {
      let mut symbol = position;
      while binomial(symbol + 1, position + 1) <= remaining {
        symbol += 1;
      }
      shifted[position] = symbol;
      remaining -= binomial(symbol, position + 1);
    }
    shifted
      .into_iter()
      .enumerate()
      .map(|(position, symbol)| symbol - self.shift(position))
      .collect()
  }

  /// Whether the word is monotone for this family: strictly increasing when
  /// repetition is forbidden, weakly increasing when it is allowed.
  pub fn is_monotone(self, word: &[usize]) -> bool {
    word.windows(2).all(|pair| match self {
      Repetition::Forbidden => pair[0] < pair[1],
      Repetition::Allowed => pair[0] <= pair[1],
    })
  }

  /// Every monotone word of length `degree` over `nsymbols` symbols, in
  /// colexicographic order, so the position in this iterator is
  /// [`Self::rank`].
  ///
  /// The successor runs on the *shifted* word, where both families are
  /// strictly increasing and the step is the same; the shift is undone on the
  /// way out. Empty when the alphabet cannot supply a word of that length,
  /// which for a forbidden repetition is any degree above `nsymbols` and for
  /// an allowed one only an empty alphabet.
  pub fn words(self, nsymbols: usize, degree: usize) -> impl Iterator<Item = Vec<usize>> {
    let shifted_nsymbols = self.shifted_nsymbols(nsymbols, degree);
    let mut shifted: Option<Vec<usize>> =
      (degree <= shifted_nsymbols).then(|| (0..degree).collect());

    std::iter::from_fn(move || {
      let current = shifted.clone()?;
      shifted = colex_successor(&current, shifted_nsymbols);
      Some(
        current
          .iter()
          .enumerate()
          .map(|(position, &symbol)| symbol - self.shift(position))
          .collect(),
      )
    })
  }
}

/// The colexicographic successor of a strictly increasing word over `nsymbols`
/// symbols, or `None` at the last.
///
/// Raise the lowest symbol that has room below its successor and reset
/// everything under it to the smallest word: colex orders by the largest
/// symbol first, so the low end is what turns over.
fn colex_successor(word: &[usize], nsymbols: usize) -> Option<Vec<usize>> {
  let ceiling = |position: usize| {
    word
      .get(position + 1)
      .copied()
      .unwrap_or(nsymbols)
      .saturating_sub(1)
  };
  let position = (0..word.len()).find(|&position| word[position] < ceiling(position))?;

  let mut next = word.to_vec();
  next[position] += 1;
  next[..position]
    .iter_mut()
    .enumerate()
    .for_each(|(index, symbol)| *symbol = index);
  Some(next)
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::{Combination, Composition, combinations};

  /// The word of a composition: its symbol repeated with the multiplicity of
  /// each part, ascending.
  fn composition_word(composition: &Composition) -> Vec<usize> {
    composition
      .parts()
      .iter()
      .enumerate()
      .flat_map(|(symbol, &multiplicity)| std::iter::repeat_n(symbol, multiplicity))
      .collect()
  }

  /// Enumeration and ranking are inverse, for both families: the position in
  /// [`Repetition::words`] is the word's [`Repetition::rank`], and the count
  /// is the number enumerated.
  #[test]
  fn rank_is_the_position_in_the_enumeration() {
    for repetition in [Repetition::Forbidden, Repetition::Allowed] {
      for nsymbols in 0..=5 {
        for degree in 0..=4 {
          let words: Vec<_> = repetition.words(nsymbols, degree).collect();
          assert_eq!(words.len(), repetition.count(nsymbols, degree));
          for (position, word) in words.iter().enumerate() {
            assert!(repetition.is_monotone(word));
            assert!(word.iter().all(|&symbol| symbol < nsymbols));
            assert_eq!(repetition.rank(word), position);
          }
        }
      }
    }
  }

  /// The forbidden family reproduces [`Combination`] exactly: same words, same
  /// order, same ranks.
  ///
  /// This is what licenses the unified implementation to replace the existing
  /// one. Order as well as content, since a rank is only meaningful against an
  /// enumeration.
  #[test]
  fn forbidden_repetition_is_the_combination() {
    for nsymbols in 0..=5 {
      for degree in 0..=4 {
        let unified: Vec<_> = Repetition::Forbidden.words(nsymbols, degree).collect();
        let existing: Vec<Vec<usize>> = combinations(nsymbols, degree)
          .map(|combination| combination.iter().collect())
          .collect();
        assert_eq!(unified, existing, "n={nsymbols} k={degree}");

        for word in &unified {
          let combination = Combination::from_increasing(word.iter().copied());
          assert_eq!(Repetition::Forbidden.rank(word), combination.rank());
        }
      }
    }
  }

  /// A rank is independent of the alphabet size: widening the alphabet leaves
  /// every existing word where it was.
  ///
  /// This is what colex is *for*, and the formula shows it -- the sum runs
  /// over the word and never mentions `nsymbols`. It holds for both families
  /// here, which is the substantive claim: the symmetric side is not a second
  /// convention that happens to agree, it is the same one.
  #[test]
  fn rank_does_not_depend_on_the_alphabet() {
    for repetition in [Repetition::Forbidden, Repetition::Allowed] {
      for degree in 0..=3 {
        for nsymbols in 0..=4 {
          for word in repetition.words(nsymbols, degree) {
            // The same word, found in a wider alphabet, keeps its position.
            for wider in nsymbols..=6 {
              let position = repetition
                .words(wider, degree)
                .position(|other| other == word);
              assert_eq!(position, Some(repetition.rank(&word)));
            }
          }
        }
      }
    }
  }

  /// The allowed family *is* [`Composition`]: same words, same order, same
  /// ranks.
  ///
  /// The counterpart of [`forbidden_repetition_is_the_combination`], and
  /// together they are the claim the module exists to make -- both families
  /// enumerated and ranked by one implementation, differing only in the shift.
  ///
  /// [`forbidden_repetition_is_the_combination`]: self::forbidden_repetition_is_the_combination
  #[test]
  fn allowed_repetition_is_the_composition() {
    for nsymbols in 1..=5 {
      for degree in 0..=4 {
        let unified: Vec<_> = Repetition::Allowed.words(nsymbols, degree).collect();
        let existing: Vec<Vec<usize>> = Composition::all(nsymbols, degree)
          .map(|composition| composition_word(&composition))
          .collect();
        assert_eq!(unified, existing, "n={nsymbols} k={degree}");

        for word in &unified {
          let composition = Composition::from_word(nsymbols, word);
          assert_eq!(Repetition::Allowed.rank(word), composition.rank());
        }
      }
    }
  }

  /// Ranking inverts the enumeration for both families, without walking it.
  #[test]
  fn word_from_rank_inverts_rank() {
    for repetition in [Repetition::Forbidden, Repetition::Allowed] {
      for nsymbols in 0..=5 {
        for degree in 0..=4 {
          for (position, word) in repetition.words(nsymbols, degree).enumerate() {
            assert_eq!(repetition.word_from_rank(nsymbols, degree, position), word);
          }
        }
      }
    }
  }

  /// Both families agree at the degenerate degrees, where there is no
  /// repetition to permit: one empty word at degree zero, and the alphabet
  /// itself at degree one.
  #[test]
  fn the_families_coincide_below_degree_two() {
    for nsymbols in 0..=5 {
      for degree in 0..=1 {
        let forbidden: Vec<_> = Repetition::Forbidden.words(nsymbols, degree).collect();
        let allowed: Vec<_> = Repetition::Allowed.words(nsymbols, degree).collect();
        assert_eq!(forbidden, allowed);
      }
    }
  }
}
