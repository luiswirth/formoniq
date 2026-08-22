//! The bijections $S_n$, colex-ranked.

use multiindex::Permutation;

/// The factorial number system: the rank $sum_j d_j dot j!$ with
/// $d_j = \#{i < j : p_i < p_j}$ is the position in [`Permutation::all`],
/// and [`Permutation::from_rank`] inverts it.
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
