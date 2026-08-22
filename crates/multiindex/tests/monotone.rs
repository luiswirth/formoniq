//! The two monotone-word families, [`Repetition::Forbidden`] (combinations)
//! and [`Repetition::Allowed`] (compositions), as one enumeration.

use multiindex::Repetition;

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

/// A rank is independent of the alphabet size: widening the alphabet leaves
/// every existing word where it was.
///
/// This is what colex is for, and the formula shows it, the sum runs
/// over the word and never mentions `nsymbols`. It holds for both families
/// here, which is the substantive claim: the symmetric side is not a second
/// convention that happens to agree: it is the same one.
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
