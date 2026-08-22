//! The [`Tensor`] product algebra: the Koszul transfer and its homotopy
//! formula, and the graded commutativity of the product across both families.

use approx::assert_relative_eq;
use multialgebra::tensor::{covariant_slots, tensor_dim};
use multialgebra::{Factor, Tensor, Vector};
use multiindex::Sign;

/// A homogeneous polynomial form $"Sym"^r times.circle Lambda^k$ with
/// deterministic components.
fn poly_form(dim: usize, r: usize, k: usize, seed: usize) -> Tensor {
  let factors = covariant_slots([Factor::symmetric(r), Factor::alternating(k)], dim);
  let len = tensor_dim(&factors);
  Tensor::new(
    factors,
    Vector::from_fn(len, |i, _| ((seed + 5 * i) % 7) as f64 - 3.0),
  )
}

/// The product is graded-commutative in the Koszul sense:
/// $b a = (-1)^(abs(a) abs(b)) a b$, where the degree that counts is the
/// alternating one, symmetric factors being even.
///
/// Checked on a mixed shape, $"Sym" times.circle Lambda$, which is where the
/// sign is a real claim: on one factor it is the wedge's antisymmetry, and on
/// a purely symmetric shape it is plain commutativity, so neither alone
/// exercises the rule.
#[test]
fn the_product_is_koszul_graded_commutative() {
  let dim = 3;
  for left in 0..=2 {
    for right in 0..=2 {
      let a = poly_form(dim, 1, left, 1);
      let b = poly_form(dim, 2, right, 2);
      let sign = Sign::from_parity(left * right).as_f64();
      assert_relative_eq!(
        a.product(&b).components(),
        &(sign * b.product(&a)).components(),
        epsilon = 1e-12
      );
    }
  }
}

/// Transferring twice in the same direction vanishes, both ways round:
/// $dif compose dif = 0$ and $kappa compose kappa = 0$.
///
/// One law for two operators, which is the point of [`Tensor::transfer`],
/// they are the same operation in opposite directions, so nilpotency is one
/// statement about it rather than two coincidences.
#[test]
fn transferring_twice_in_one_direction_vanishes() {
  for dim in 1..=4 {
    for r in 0..=3 {
      for k in 0..=dim {
        let form = poly_form(dim, r, k, 1);
        // Sym -> Lambda twice: the exterior derivative.
        let twice = form.transfer(0, 1).transfer(0, 1);
        assert_relative_eq!(twice.components().amax(), 0.0, epsilon = 1e-12);
        // Lambda -> Sym twice: the Koszul operator.
        let twice = form.transfer(1, 0).transfer(1, 0);
        assert_relative_eq!(twice.components().amax(), 0.0, epsilon = 1e-12);
      }
    }
  }
}

/// The Koszul homotopy formula: on homogeneous $"Sym"^r times.circle
/// Lambda^k$, $dif kappa + kappa dif = (r + k) id$.
///
/// The identity the whole polynomial de Rham complex rests on: it is what
/// makes that complex exact, and hence what the trimmed spaces
/// $P^-_r Lambda^k$ are cut out by. Checking it here checks that both
/// directions of the transfer carry the right signs and the right
/// multiplicities, which no weaker law does: nilpotency alone passes on an
/// operator scaled by anything.
#[test]
fn the_koszul_homotopy_formula_holds() {
  for dim in 1..=4 {
    for r in 0..=3 {
      for k in 0..=dim {
        let form = poly_form(dim, r, k, 2);
        let dif_then_koszul = form.transfer(0, 1).transfer(1, 0);
        let koszul_then_dif = form.transfer(1, 0).transfer(0, 1);
        let sum = dif_then_koszul + koszul_then_dif;
        let expected = (r + k) as f64 * form.clone();
        assert_relative_eq!(sum.components(), expected.components(), epsilon = 1e-9);
        if r + k > 0 {
          assert!(
            expected.components().amax() > 0.0,
            "the law would hold vacuously"
          );
        }
      }
    }
  }
}
