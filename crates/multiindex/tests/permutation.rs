//! The bijections $S_n$, colex-ranked.

use multiindex::{Permutation, factorial};

/// The checked constructor decides bijectivity: it accepts a one-line word
/// and rejects the two ways of failing, an entry out of range and an entry
/// that repeats.
///
/// Both directions, so that a constructor returning `Some` unconditionally
/// would fail. Every permutation the enumeration produces is valid, which is
/// what says the generator and the predicate agree.
#[test]
fn new_checked_decides_bijectivity() {
  for n in 0..=5 {
    for p in Permutation::all(n) {
      assert!(p.is_valid());
      assert!(Permutation::new_checked(p.into_parts()).is_some());
    }
  }
  assert!(Permutation::new_checked([0, 1, 3]).is_none());
  assert!(Permutation::new_checked([0, 1, 1]).is_none());
}

/// The frozen enumeration order, stated explicitly rather than derived, so a
/// change to the generator is a test failure and not a silent renumbering.
#[test]
fn colex_order_is_frozen() {
  let s3: Vec<Vec<usize>> = Permutation::all(3).map(Permutation::into_parts).collect();
  assert_eq!(
    s3,
    vec![
      vec![2, 1, 0],
      vec![1, 2, 0],
      vec![2, 0, 1],
      vec![0, 2, 1],
      vec![1, 0, 2],
      vec![0, 1, 2],
    ]
  );
}

/// Colex is lex on the reversed word, the defining property.
#[test]
fn colex_is_lex_on_reversed_word() {
  for n in 0..=6 {
    let reversed: Vec<Vec<usize>> = Permutation::all(n)
      .map(|p| p.iter().rev().collect())
      .collect();
    let mut sorted = reversed.clone();
    sorted.sort();
    assert_eq!(reversed, sorted, "n = {n}");
  }
}

#[test]
fn all_is_complete_and_distinct() {
  for n in 0..=6 {
    let all: Vec<Permutation> = Permutation::all(n).collect();
    assert_eq!(all.len(), factorial(n));
    let mut distinct = all.clone();
    distinct.sort();
    distinct.dedup();
    assert_eq!(distinct.len(), factorial(n));
  }
}

#[test]
fn rank_is_position_in_all() {
  for n in 0..=6 {
    for (position, p) in Permutation::all(n).enumerate() {
      assert_eq!(p.rank(), position, "n = {n}");
      assert_eq!(Permutation::from_rank(n, position), p);
    }
  }
}

/// The rank formula carries no $n$: $S_n$ is an initial segment of $S_(n+1)$
/// under the embedding that raises every value and appends $0$.
#[test]
fn rank_is_independent_of_length() {
  for n in 0..=5 {
    for p in Permutation::all(n) {
      let embedded: Permutation = p.iter().map(|v| v + 1).chain(std::iter::once(0)).collect();
      assert_eq!(embedded.rank(), p.rank());
    }
  }
}

#[test]
fn inverse_and_composition_are_a_group() {
  for n in 0..=5 {
    let id = Permutation::identity(n);
    for p in Permutation::all(n) {
      assert_eq!(p.compose(&p.inverse()), id);
      assert_eq!(p.inverse().compose(&p), id);
      assert_eq!(p.inverse().inverse(), p);
    }
  }
}

/// $"sgn"$ is a homomorphism $S_n -> {plus.minus 1}$.
#[test]
fn sign_is_a_homomorphism() {
  for n in 0..=4 {
    for p in Permutation::all(n) {
      for q in Permutation::all(n) {
        assert_eq!(p.compose(&q).sign(), p.sign() * q.sign());
      }
      assert_eq!(p.inverse().sign(), p.sign());
    }
  }
}
