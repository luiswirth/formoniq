//! Functoriality of a single [`Factor`], on either symmetry.

use approx::assert_relative_eq;
use multialgebra::{Degree, Factor, Matrix, Symmetry, tensor};
use multiindex::{Composition, binomial};

/// A deterministic matrix with no symmetry, so a transposed index or a
/// dropped factor cannot pass unnoticed.
fn probe(nrows: usize, ncols: usize, seed: usize) -> Matrix {
  Matrix::from_fn(nrows, ncols, |i, j| {
    ((7 * i + 3 * j + 5 * seed + 1) % 11) as f64 - 5.0
  })
}

/// $dim Lambda^k (RR^n) = binom(n,k)$ and $dim "Sym"^k (RR^n) =
/// binom(n+k-1,k)$, and the two agree exactly at degree $0$ and $1$, where
/// $Lambda^1 = "Sym"^1 = V$ and there is no symmetry to impose.
#[test]
fn factor_dimensions_and_their_degenerate_agreement() {
  for n in 0..=4 {
    for k in 0..=4 {
      assert_eq!(Factor::alternating(k).multidim(n), binomial(n, k));
      assert_eq!(Factor::symmetric(k).multidim(n), Composition::count(n, k));
      if k <= 1 {
        assert_eq!(
          Factor::alternating(k).multidim(n),
          Factor::symmetric(k).multidim(n)
        );
      }
    }
    // Only the alternating side has a top degree.
    assert_eq!(Factor::alternating(n + 1).multidim(n), 0);
    assert!(Factor::symmetric(n + 1).multidim(n) > 0 || n == 0);
    // Both are trivial below zero.
    assert_eq!(Factor::alternating(-1).multidim(n), 0);
    assert_eq!(Factor::symmetric(-1).multidim(n), 0);
  }
}

/// $F(A B) = F(A) F(B)$ for a single factor of either symmetry: Cauchy-Binet
/// on the alternating side and its permanental counterpart on the symmetric
/// one.
///
/// Swept over rectangular shapes, so the two dimensions of the map are not
/// allowed to coincide and hide a transpose.
#[test]
fn each_factor_is_a_functor() {
  for degree in 0..=3 {
    for symmetry in [Symmetry::Alternating, Symmetry::Symmetric] {
      for &(p, q, r) in &[(2, 3, 2), (3, 2, 3), (4, 3, 2), (2, 2, 4)] {
        let factor = Factor::new(symmetry, degree);
        let (a, b) = (probe(p, q, 1), probe(q, r, 2));
        assert_relative_eq!(
          factor.induced(&(&a * &b)),
          factor.induced(&a) * factor.induced(&b),
          epsilon = 1e-9
        );
      }
    }
  }
}

/// The induced map on a tensor product is the Kronecker product of the
/// per-factor ones, and is itself a functor.
///
/// This is the law the crate exists for: one composition rule covering
/// $Lambda^k times.circle "Sym"^l$ with the symmetry consulted only per
/// factor. Mixed symmetries and unequal degrees, so neither can stand in for
/// the other.
#[test]
fn the_tensor_product_of_factors_is_a_functor() {
  let factor_lists: [Vec<Factor>; 5] = [
    vec![],
    vec![Factor::alternating(2)],
    vec![Factor::symmetric(1), Factor::alternating(2)],
    vec![Factor::symmetric(2), Factor::symmetric(1)],
    vec![
      Factor::symmetric(2),
      Factor::alternating(1),
      Factor::symmetric(1),
    ],
  ];
  for factors in &factor_lists {
    for &(p, q, r) in &[(3, 3, 3), (4, 3, 3), (3, 4, 2)] {
      // The slots name the domain; a transport reads both ends off the map.
      let slots = tensor::covariant_slots(factors.iter().copied(), q);
      let functor = |map: &Matrix| tensor::Transport::new(&slots, map).to_matrix();
      let (a, b) = (probe(p, q, 3), probe(q, r, 4));
      let composed = functor(&(&a * &b));
      let separate = functor(&a) * functor(&b);

      let expected_rows: usize = factors.iter().map(|f| f.multidim(p)).product();
      let expected_cols: usize = factors.iter().map(|f| f.multidim(r)).product();
      assert_eq!(
        (composed.nrows(), composed.ncols()),
        (expected_rows, expected_cols)
      );
      assert_relative_eq!(composed, separate, epsilon = 1e-6);
    }
  }
}

/// At degree one every factor is the space itself, so the functor is the map
/// back again, whatever the symmetry. The base case that pins the conventions:
/// it fails on a transposed index or a wrong basis order.
#[test]
fn degree_one_induces_the_map_itself() {
  for symmetry in [Symmetry::Alternating, Symmetry::Symmetric] {
    let map = probe(3, 4, 5);
    let factor = Factor::new(symmetry, Degree::ONE);
    assert_relative_eq!(factor.induced(&map), map);
  }
}
