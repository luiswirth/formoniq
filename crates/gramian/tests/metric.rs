//! The metric-dependent laws of the exterior algebra.
//!
//! They live here rather than in `multialgebra` because they need a metric, and
//! that is the whole content of the crate split: the wedge and the contraction
//! are stated one crate down, where no metric exists to leak in.

use approx::assert_relative_eq;
use gramian::tensor::{TensorExt, inner};
use gramian::{Gramian, Metric};
use multialgebra::tensor::{pairing, wedge_pairing};
use multialgebra::{Matrix, Tensor, Variance, Vector, exterior_bases, exterior_dim};
use multiindex::Sign;

fn probe_matrix(nrows: usize, ncols: usize, seed: usize) -> Matrix {
  Matrix::from_fn(nrows, ncols, |i, j| {
    ((seed + 3 * i + 7 * j) % 5) as f64 / 5.0 + if i == j { 1.0 } else { 0.0 }
  })
}

fn probe_element(dim: usize, grade: usize, seed: usize, variance: Variance) -> Tensor {
  Tensor::new(
    Tensor::one_alternating(grade, variance, dim),
    Vector::from_fn(exterior_dim(dim, grade), |i, _| {
      ((seed + 5 * i) % 7) as f64 - 3.0
    }),
  )
}

fn probe_metric(dim: usize) -> Metric {
  let a = probe_matrix(dim, dim, 5);
  Metric::new(Gramian::new(
    a.transpose() * &a + Matrix::identity(dim, dim),
  ))
}
fn probe_pseudo_metric(dim: usize, q: usize) -> Metric {
  let j = Matrix::from_fn(dim, dim, |i, jj| {
    if i == jj {
      1.0
    } else if i > jj {
      ((3 * i + 5 * jj) % 4) as f64 / 8.0
    } else {
      0.0
    }
  });
  Metric::new(Gramian::pseudo_euclidean(dim - q, q).pullback(&j))
}

/// $star star = (-1)^(k(n-k)) (-1)^q$ on any signature, for both variances.
#[test]
fn hodge_star_involution() {
  for dim in 1..=4 {
    for q in 0..=dim {
      for metric in [
        Metric::new(Gramian::pseudo_euclidean(dim - q, q)),
        probe_pseudo_metric(dim, q),
      ] {
        for grade in 0..=dim {
          let sign = Sign::from_parity(grade * (dim - grade)) * Sign::from_parity(q);

          let form = probe_element(dim, grade, 2, Variance::Covariant);
          let twice = form.star(&metric, Sign::Pos).star(&metric, Sign::Pos);
          assert_relative_eq!(
            twice.components(),
            &(sign.as_f64() * form).components(),
            epsilon = 1e-12
          );

          let vector = probe_element(dim, grade, 3, Variance::Contravariant);
          let twice = vector.star(&metric, Sign::Pos).star(&metric, Sign::Pos);
          assert_relative_eq!(
            twice.components(),
            &(sign.as_f64() * vector).components(),
            epsilon = 1e-12
          );
        }
      }
    }
  }
}

/// $alpha wedge star beta = inner(alpha, beta) vol$: the defining property,
/// tying wedge, inner product and star together on every signature.
#[test]
fn wedge_with_star_is_inner_times_volume() {
  for dim in 1..=4 {
    for q in 0..=dim {
      for metric in [
        Metric::new(Gramian::pseudo_euclidean(dim - q, q)),
        probe_pseudo_metric(dim, q),
      ] {
        for grade in 0..=dim {
          let alpha = probe_element(dim, grade, 3, Variance::Covariant);
          let beta = probe_element(dim, grade, 4, Variance::Covariant);
          let wedge = alpha.wedge(&beta.star(&metric, Sign::Pos));
          assert_eq!(wedge.grade(), dim);
          assert_relative_eq!(
            wedge[0],
            inner(&alpha, &beta, &metric) * metric.det_sqrt(),
            epsilon = 1e-12
          );
        }
      }
    }
  }
}

/// For the Euclidean metric the star is the signed complement of each blade.
#[test]
fn hodge_star_euclidean_is_signed_complement() {
  for dim in 1..=4 {
    let euclidean = Metric::euclidean(dim);
    for grade in 0..=dim {
      for blade in exterior_bases(dim, grade) {
        let element = Tensor::from_blade_signed(dim, Sign::Pos, blade, Variance::Covariant);
        let star = element.star(&euclidean, Sign::Pos);
        let (sign, complement) = blade.complement_signed(dim);
        let expected = Tensor::from_blade_signed(dim, sign, complement, Variance::Covariant);
        assert_relative_eq!(star.components(), expected.components());
      }
    }
  }
}

/// The musical isomorphisms are inverse and turn the pairing into the inner
/// product.
#[test]
fn musical_isomorphisms() {
  for dim in 1..=4 {
    let metric = probe_metric(dim);
    for grade in 0..=dim {
      let v = probe_element(dim, grade, 1, Variance::Contravariant);
      let w = probe_element(dim, grade, 2, Variance::Contravariant);
      assert_relative_eq!(
        v.musical(&metric).musical(&metric).components(),
        v.components(),
        epsilon = 1e-12
      );
      assert_relative_eq!(
        pairing(&v.musical(&metric), &w),
        inner(&v, &w, &metric),
        epsilon = 1e-12
      );
    }
  }
}

/// Writing a tensor out on the full basis of $V^(times.circle d)$ and taking
/// the Euclidean dot product there is the packed inner product, scaled by
/// $product_i k_i !$.
///
/// One law over both families, and it is what says the dense embedding and
/// [`Factor::gramian`] agree on the normalization. The alternating basis is
/// orthonormal while the symmetric one has $norm(x^alpha)^2 = alpha!$, and both
/// come out right only because the embedding is unnormalized in the same sense
/// the Gramian is: a mismatch shows up here as a factorial.
#[test]
fn the_dense_embedding_matches_the_gramian_up_to_the_factorials() {
  for dim in 1..=3 {
    let metric = Metric::euclidean(dim);
    for degree in 0..=3 {
      for symmetry in [
        multialgebra::Symmetry::Alternating,
        multialgebra::Symmetry::Symmetric,
      ] {
        let slots = multialgebra::tensor::uniform_slots(
          [multialgebra::Factor::new(symmetry, degree)],
          Variance::Covariant,
          dim,
        );
        let len = multialgebra::tensor::tensor_dim(&slots);
        if len == 0 {
          continue;
        }
        let build = |seed: usize| {
          Tensor::new(
            slots.clone(),
            Vector::from_fn(len, |i, _| ((seed + 5 * i) % 7) as f64 - 3.0),
          )
        };
        let (a, b) = (build(2), build(3));
        let scale = multiindex::factorial(degree) as f64;

        assert_relative_eq!(
          a.to_free().components().dot(b.to_free().components()),
          scale * inner(&a, &b, &metric),
          epsilon = 1e-10
        );
      }
    }
  }
}

/// The three dualities of the exterior algebra, and what each of them needs.
///
/// The wedge pairing is metric-free: it asks only for a top grade to land in.
/// The Hodge star is what turns it into the inner product,
/// $angle.l alpha, star beta angle.r_wedge = inner(alpha, beta) vol$, and that
/// step is exactly where the metric enters. Stating the two together is what
/// keeps them from being conflated.
#[test]
fn the_star_is_what_turns_the_wedge_pairing_into_the_inner_product() {
  for dim in 1..=4 {
    for q in 0..=dim {
      let metric = Metric::new(Gramian::pseudo_euclidean(dim - q, q));
      let volume = metric.det_sqrt();
      for grade in 0..=dim {
        let alpha = probe_element(dim, grade, 3, Variance::Covariant);
        let beta = probe_element(dim, grade, 4, Variance::Covariant);

        assert_relative_eq!(
          wedge_pairing(&alpha, &beta.star(&metric, Sign::Pos)),
          inner(&alpha, &beta, &metric) * volume,
          epsilon = 1e-10
        );
      }
    }
  }
}

/// A free slot has no Hodge star: there is a top degree only after a quotient
/// by the sign character, and the free power has none.
#[test]
#[should_panic(expected = "no top degree")]
fn a_free_slot_has_no_hodge_star() {
  use multialgebra::{Factor, Symmetry, tensor::uniform_slots};

  let dim = 3;
  let slots = uniform_slots([Factor::new(Symmetry::Free, 2)], Variance::Covariant, dim);
  let tensor = Tensor::zero(slots);
  let _ = tensor.star(&Metric::euclidean(dim), Sign::Pos);
}
