//! The metric-dependent laws of the exterior algebra.
//!
//! They live here rather than in `multialgebra` because they need a metric, and
//! that is the whole content of the crate split: the wedge and the contraction
//! are stated one crate down, where no metric exists to leak in.

use approx::assert_relative_eq;
use metric::Metric;
use metric::tensor::{TensorExt, inner};
use multialgebra::tensor::{one_alternating, pairing};
use multialgebra::{Matrix, Tensor, Variance, Vector, exterior_dim};
use multiindex::Sign;

fn probe_matrix(nrows: usize, ncols: usize, seed: usize) -> Matrix {
  Matrix::from_fn(nrows, ncols, |i, j| {
    ((seed + 3 * i + 7 * j) % 5) as f64 / 5.0 + if i == j { 1.0 } else { 0.0 }
  })
}

fn probe_element(dim: usize, grade: usize, seed: usize, variance: Variance) -> Tensor {
  Tensor::new(
    one_alternating(grade, variance, dim),
    Vector::from_fn(exterior_dim(dim, grade), |i, _| {
      ((seed + 5 * i) % 7) as f64 - 3.0
    }),
  )
}

fn probe_metric(dim: usize) -> Metric {
  let a = probe_matrix(dim, dim, 5);
  Metric::new(
    Variance::Covariant,
    a.transpose() * &a + Matrix::identity(dim, dim),
  )
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
  Metric::pseudo_euclidean(dim - q, q).pullback(&j)
}

/// $star star = (-1)^(k(n-k)) (-1)^q$ on any signature, for both variances.
#[test]
fn hodge_star_involution() {
  for dim in 1..=4 {
    for q in 0..=dim {
      for metric in [
        Metric::pseudo_euclidean(dim - q, q),
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
        Metric::pseudo_euclidean(dim - q, q),
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

/// Sylvester's law of inertia: the signature is invariant under congruence,
/// that is under pullback along any invertible map.
#[test]
fn signature_is_congruence_invariant() {
  for dim in 1..=4 {
    for q in 0..=dim {
      let g = Metric::pseudo_euclidean(dim - q, q);
      assert_eq!(
        g.pullback(&probe_matrix(dim, dim, 5)).signature(),
        (dim - q, q)
      );
    }
  }
}
