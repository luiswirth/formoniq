//! Grundmann-Möller quadrature: exactness against the closed-form integral
//! of a barycentric monomial, in every dimension.

use approx::assert_abs_diff_eq;
use itertools::Itertools;
use multiindex::{Composition, factorial_f64};
use simplicial::Dim;
use simplicial::atlas::{BaryRef, SimplexQuadRule, unit_simplex_volume};

/// The exact integral of a barycentric monomial over the unit simplex:
/// $integral_Delta lambda^alpha = n! alpha_0 ! dots.c alpha_n ! /
/// (n + |alpha|)! dot vol$.
fn exact_barycentric_monomial(alpha: &[usize]) -> f64 {
  let n = alpha.len() - 1;
  let total: usize = alpha.iter().sum();
  let numerator: f64 = factorial_f64(n) * alpha.iter().map(|&a| factorial_f64(a)).product::<f64>();
  numerator / factorial_f64(n + total) * unit_simplex_volume(n)
}

/// Grundmann-Möller integrates every barycentric monomial of degree
/// <= 2s + 1 exactly, in every dimension.
#[test]
fn grundmann_moeller_is_exact_on_polynomials() {
  for dim in (0..=4usize).map(Dim::from) {
    for s in 0..=3 {
      let quadrule = SimplexQuadRule::grundmann_moeller(dim, s);
      let max_degree = 2 * s + 1;
      for degree in 0..=max_degree {
        for alpha in Composition::all((dim + 1).index(), degree) {
          let monomial = |bary: BaryRef| -> f64 {
            (0..=dim.index())
              .map(|i| bary[i].powi(alpha.parts()[i] as i32))
              .product()
          };
          let computed = quadrule.integrate_unit(&monomial, unit_simplex_volume(dim));
          let exact = exact_barycentric_monomial(alpha.parts());
          assert_abs_diff_eq!(computed, exact, epsilon = 1e-12);
        }
      }
    }
  }
}

/// The nodes are barycentric: their weights sum to one, and so do the
/// quadrature weights.
#[test]
fn nodes_and_weights_are_barycentric() {
  for dim in (0..=4usize).map(Dim::from) {
    for s in 0..=3 {
      let quadrule = SimplexQuadRule::grundmann_moeller(dim, s);
      assert_abs_diff_eq!(quadrule.weights().sum(), 1.0, epsilon = 1e-12);
      for bary in quadrule.points() {
        assert_abs_diff_eq!(bary.view().sum(), 1.0, epsilon = 1e-12);
      }
    }
  }
}

/// The degree constructor guarantees the requested exactness.
#[test]
fn degree_constructor_is_exact() {
  for dim in (1..=3usize).map(Dim::from) {
    for degree in 0..=5 {
      let quadrule = SimplexQuadRule::degree(dim, degree);
      let monomial = |bary: BaryRef| bary[1].powi(degree as i32);
      let alpha: Vec<usize> = std::iter::once(0)
        .chain(std::iter::once(degree))
        .chain(std::iter::repeat_n(0, dim.index() - 1))
        .collect_vec();
      let computed = quadrule.integrate_unit(&monomial, unit_simplex_volume(dim));
      let exact = exact_barycentric_monomial(&alpha);
      assert_abs_diff_eq!(computed, exact, epsilon = 1e-12);
    }
  }
}
