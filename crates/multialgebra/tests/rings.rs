//! The algebra over a general commutative ring, stated exactly.
//!
//! Over an exact ring the laws are equalities rather than tolerances, and that
//! is how they are written here: no epsilon appears in this file. The ring is
//! a sweep axis of its own. `i64` reaches every operation that does not
//! dualize a slot; the reciprocal basis of a symmetric slot is $x^alpha \/
//! alpha!$, which over the integers spans the divided power algebra rather
//! than the symmetric one, so the operations that land there ask for a
//! `RationalAlgebra`. `f64` and `Complex64` are the rings the rest of the
//! workspace runs on, and they are exact here too, not by luck: the fixtures
//! are small integers and the operations are sums of products of them.

use multialgebra::{
  Factor, Matrix, RationalAlgebra, Ring, Symmetry, Tensor, Variance, Vector, determinant,
  exterior_dim, exterior_power, from_integer, symmetric_power,
  tensor::{covariant_slots, one_alternating, pairing, uniform_slots},
};
use num_complex::Complex64;
use num_rational::Rational64;

/// A deterministic matrix with no symmetry, so a transposed index or a dropped
/// factor cannot pass unnoticed. Integer-valued, which is what lets the same
/// fixture serve every ring.
fn probe<R: Ring>(nrows: usize, ncols: usize, seed: usize) -> Matrix<R> {
  Matrix::from_fn(nrows, ncols, |i, j| {
    from_integer(((7 * i + 3 * j + 5 * seed + 1) % 11) as i64 - 5)
  })
}

fn probe_element<R: Ring>(dim: usize, grade: usize, seed: usize, variance: Variance) -> Tensor<R> {
  Tensor::new(
    one_alternating(grade, variance, dim),
    Vector::from_fn(exterior_dim(dim, grade), |i, _| {
      from_integer(((seed + 5 * i) % 7) as i64 - 3)
    }),
  )
}

fn probe_symmetric<R: Ring>(dim: usize, degree: usize, seed: usize) -> Tensor<R> {
  let slots = covariant_slots([Factor::symmetric(degree)], dim);
  let count = Factor::symmetric(degree).multidim(dim);
  Tensor::new(
    slots,
    Vector::from_fn(count, |i, _| from_integer(((seed + 3 * i) % 5) as i64 - 2)),
  )
}

/// Cauchy-Binet, $Lambda^k (A B) = (Lambda^k A)(Lambda^k B)$: functoriality of
/// the exterior power, as an exact identity of integer matrices.
///
/// The minors are integer polynomials in the entries, so this is where the
/// ring-level determinant earns its keep: `nalgebra`'s asks for a field
/// because it eliminates, and elimination divides.
fn exterior_power_is_functorial_over<R: Ring>() {
  for inner in 0..=3 {
    for mid in 0..=3 {
      for outer in 0..=3 {
        let (a, b) = (probe::<R>(outer, mid, 1), probe::<R>(mid, inner, 2));
        for k in 0..=mid.max(outer).max(inner) {
          assert!(
            exterior_power(&(&a * &b), k) == exterior_power(&a, k) * exterior_power(&b, k),
            "Lambda^{k} of a {outer}x{mid}x{inner} composite"
          );
        }
      }
    }
  }
}

/// The permanental counterpart on the symmetric side,
/// $"Sym"^k (A B) = ("Sym"^k A)("Sym"^k B)$: the same law read through the
/// other character, and equally exact.
fn symmetric_power_is_functorial_over<R: Ring>() {
  for inner in 0..=3 {
    for mid in 0..=3 {
      for outer in 0..=3 {
        let (a, b) = (probe::<R>(outer, mid, 3), probe::<R>(mid, inner, 5));
        for k in 0..=3 {
          assert!(
            symmetric_power(&(&a * &b), k) == symmetric_power(&a, k) * symmetric_power(&b, k),
            "Sym^{k} of a {outer}x{mid}x{inner} composite"
          );
        }
      }
    }
  }
}

/// $iota_v^2 = 0$: the interior product is an antiderivation of square zero.
///
/// A vanishing law is the kind that passes for the wrong reason, so it is
/// paired below with the check that the first contraction does not already
/// vanish.
fn the_interior_product_is_nilpotent_over<R: Ring>() {
  for dim in 1..=4 {
    for grade in 0..=dim {
      let form = probe_element::<R>(dim, grade, 1, Variance::Covariant);
      let v = probe_element::<R>(dim, 1, 2, Variance::Contravariant);
      let once = form.interior_product(&v);
      let twice = once.interior_product(&v);
      assert!(twice.components().iter().all(|c| *c == R::zero()));
      if grade >= 2 {
        assert!(
          once.components().iter().any(|c| *c != R::zero()),
          "dim {dim} grade {grade}: the first contraction already vanished, so \
           the nilpotency above says nothing"
        );
      }
    }
  }
}

/// The wedge is associative and Koszul graded-commutative,
/// $alpha wedge beta = (-1)^(k l) beta wedge alpha$.
fn the_wedge_is_a_graded_algebra_over<R: Ring>() {
  for dim in 0..=4 {
    for k in 0..=dim {
      for l in 0..=dim {
        let (a, b) = (
          probe_element::<R>(dim, k, 1, Variance::Covariant),
          probe_element::<R>(dim, l, 2, Variance::Covariant),
        );
        let sign: R = from_integer(if (k * l) % 2 == 0 { 1 } else { -1 });
        assert!(
          a.wedge(&b) == b.wedge(&a) * sign,
          "dim {dim} grades {k},{l}"
        );

        for m in 0..=dim {
          let c = probe_element::<R>(dim, m, 3, Variance::Covariant);
          assert!(a.wedge(&b).wedge(&c) == a.wedge(&b.wedge(&c)));
        }
      }
    }
  }
}

/// Adjointness of the transport under the duality pairing,
/// $chevron.l A^* omega, v chevron.r = chevron.l omega, A_* v chevron.r$.
///
/// The pairing is bilinear and metric-free, and the pullback is the adjoint of
/// the pushforward, so this is the law that pins the reciprocal basis. It
/// dualizes, hence the `RationalAlgebra` bound, and it is swept over the
/// symmetric family as well as the alternating one, where every factorial is
/// $1$ and the law would say nothing about the change of basis.
fn the_transport_is_adjoint_over<R: RationalAlgebra>() {
  for source in 1..=3 {
    for target in 1..=3 {
      let map = probe::<R>(target, source, 4);
      for degree in 0..=2 {
        for symmetry in [Symmetry::Alternating, Symmetry::Symmetric] {
          let shape =
            |dim| uniform_slots([Factor::new(symmetry, degree)], Variance::Covariant, dim);
          let count = Factor::new(symmetry, degree).multidim(target);
          let form: Tensor<R> = Tensor::new(
            shape(target),
            Vector::from_fn(count, |i, _| from_integer(((3 * i + 1) % 7) as i64 - 3)),
          );
          let vector: Tensor<R> = Tensor::new(
            uniform_slots(
              [Factor::new(symmetry, degree)],
              Variance::Contravariant,
              source,
            ),
            Vector::from_fn(Factor::new(symmetry, degree).multidim(source), |i, _| {
              from_integer(((5 * i + 2) % 9) as i64 - 4)
            }),
          );

          assert!(
            pairing(&form.pullback(&map), &vector) == pairing(&form, &vector.pushforward(&map)),
            "{symmetry:?} degree {degree}, {target} <- {source}"
          );
        }
      }
    }
  }
}

/// Evaluation of a symmetric factor divides by $r!$, so it is the other
/// operation that needs the factorials inverted, and over $QQ$ it is exact.
fn evaluating_a_symmetric_factor_is_the_monomial_over<R: RationalAlgebra>() {
  for dim in 1..=3 {
    for degree in 0..=3 {
      let poly = probe_symmetric::<R>(dim, degree, 1);
      let point = probe_element::<R>(dim, 1, 2, Variance::Contravariant);
      let value = poly.evaluate(0, &point);
      assert_eq!(value.slots().len(), 0, "evaluation consumes the factor");
      // Homogeneity: p(2 x) = 2^r p(x), which fails for any misplaced r!.
      let two: R = from_integer(2);
      let scaled = poly.evaluate(0, &(point * two.clone()));
      let factor = (0..degree).fold(R::one(), |acc, _| acc * two.clone());
      assert!(scaled.as_scalar() == value.as_scalar() * factor);
    }
  }
}

/// Extension of scalars is natural: $phi(alpha wedge beta) = phi alpha wedge
/// phi beta$, and likewise for the interior product and the pushforward.
///
/// This is the law a hardcoded coefficient breaks. It holds because every
/// structure constant of the algebra is the image of an integer, and a ring
/// map fixes those, so it is a statement about the implementation and not only
/// about the mathematics.
#[test]
fn extension_of_scalars_is_natural() {
  let phi = |n: &i64| Rational64::from_integer(*n);
  for dim in 0..=4 {
    for k in 0..=dim {
      for l in 0..=dim {
        let a = probe_element::<i64>(dim, k, 1, Variance::Covariant);
        let b = probe_element::<i64>(dim, l, 2, Variance::Covariant);
        assert!(
          a.wedge(&b).extend_scalars(phi) == a.extend_scalars(phi).wedge(&b.extend_scalars(phi))
        );
      }

      let form = probe_element::<i64>(dim, k, 3, Variance::Contravariant);
      let v = probe_element::<i64>(dim, 1, 4, Variance::Covariant);
      assert!(
        form.interior_product(&v).extend_scalars(phi)
          == form
            .extend_scalars(phi)
            .interior_product(&v.extend_scalars(phi))
      );

      let map = probe::<i64>(dim, dim, 5);
      assert!(
        form.pushforward(&map).extend_scalars(phi)
          == form.extend_scalars(phi).pushforward(&map.map(|n| phi(&n)))
      );
    }
  }
}

#[test]
fn exterior_power_is_functorial() {
  exterior_power_is_functorial_over::<i64>();
  exterior_power_is_functorial_over::<Rational64>();
  exterior_power_is_functorial_over::<f64>();
  exterior_power_is_functorial_over::<Complex64>();
}

#[test]
fn symmetric_power_is_functorial() {
  symmetric_power_is_functorial_over::<i64>();
  symmetric_power_is_functorial_over::<Rational64>();
  symmetric_power_is_functorial_over::<f64>();
  symmetric_power_is_functorial_over::<Complex64>();
}

#[test]
fn the_interior_product_is_nilpotent() {
  the_interior_product_is_nilpotent_over::<i64>();
  the_interior_product_is_nilpotent_over::<Rational64>();
  the_interior_product_is_nilpotent_over::<f64>();
  the_interior_product_is_nilpotent_over::<Complex64>();
}

#[test]
fn the_wedge_is_a_graded_algebra() {
  the_wedge_is_a_graded_algebra_over::<i64>();
  the_wedge_is_a_graded_algebra_over::<Rational64>();
  the_wedge_is_a_graded_algebra_over::<f64>();
  the_wedge_is_a_graded_algebra_over::<Complex64>();
}

#[test]
fn the_transport_is_adjoint() {
  the_transport_is_adjoint_over::<Rational64>();
  the_transport_is_adjoint_over::<f64>();
  the_transport_is_adjoint_over::<Complex64>();
}

#[test]
fn evaluating_a_symmetric_factor_is_the_monomial() {
  evaluating_a_symmetric_factor_is_the_monomial_over::<Rational64>();
}

/// $det A = Lambda^n A$: the top exterior power of an endomorphism of an
/// $n$-dimensional space is the one-by-one matrix of its determinant.
///
/// The determinant is not a separate construction but the extremal case of
/// the functor, which is what makes Cauchy-Binet at $k = n$ the
/// multiplicativity $det(A B) = det A dot det B$.
#[test]
fn the_determinant_is_the_top_exterior_power() {
  fn check<R: Ring>() {
    for n in 0..=4 {
      let a = probe::<R>(n, n, 7);
      let top = exterior_power(&a, n);
      assert_eq!((top.nrows(), top.ncols()), (1, 1));
      assert!(top[(0, 0)] == determinant(&a), "dim {n}");
    }
  }
  check::<i64>();
  check::<Rational64>();
  check::<f64>();
  check::<Complex64>();
}
