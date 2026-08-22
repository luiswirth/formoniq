//! The characterization of an affine map: it is exactly a map preserving the
//! affine combination, which is the only combination of points an affine space
//! admits.

use coorder::{CoordSpace, Coords, Matrix, Vector, affine::AffineTransform};

enum Source {}
impl CoordSpace for Source {
  const NAME: &'static str = "source";
}
enum Target {}
impl CoordSpace for Target {
  const NAME: &'static str = "target";
}

fn close(a: &Vector, b: &Vector) {
  assert_eq!(a.len(), b.len());
  assert!((a - b).norm() < 1e-9, "{a:?} != {b:?}");
}

/// A deterministic full-column-rank `nrows x ncols` matrix (ncols <= nrows):
/// unit lower-triangular columns, echelon and hence injective.
fn full_col_rank(nrows: usize, ncols: usize) -> Matrix {
  Matrix::from_fn(nrows, ncols, |i, j| {
    if i == j {
      1.0
    } else if i > j {
      0.5
    } else {
      0.0
    }
  })
}

/// The `k`-th of a family of distinct points of the given dimension.
fn point<S: CoordSpace>(dim: usize, k: usize) -> Coords<S> {
  Coords::new(Vector::from_fn(dim, |i, _| {
    2.0 - 0.3 * i as f64 + 1.7 * k as f64 * (1.0 + i as f64)
  }))
}

/// Weights summing to one, so an affine combination of `n` points is defined.
fn weights(n: usize) -> Vec<f64> {
  let raw: Vec<f64> = (0..n).map(|i| 1.0 + 0.5 * i as f64).collect();
  let total: f64 = raw.iter().sum();
  raw.iter().map(|w| w / total).collect()
}

/// The characterization of an affine map: it commutes with affine combinations,
/// $T(sum_i lambda_i p_i) = sum_i lambda_i T(p_i)$. Linear maps satisfy this for
/// all weights, affine ones exactly for those summing to one.
#[test]
fn affine_maps_preserve_affine_combinations() {
  for image in 0..=4 {
    for domain in 0..=image {
      let translation: Coords<Target> = point(image, 7);
      let t = AffineTransform::new(translation, full_col_rank(image, domain));
      for npoints in 1..=4 {
        let points: Vec<Coords<Source>> = (0..npoints).map(|k| point(domain, k)).collect();
        let weights = weights(npoints);

        let mapped: Vec<Coords<Target>> = points
          .iter()
          .map(|p| t.apply_forward(p.as_view()))
          .collect();

        close(
          t.apply_forward(
            Coords::affine_combination(weights.iter().copied().zip(points.iter())).as_view(),
          )
          .vector(),
          Coords::affine_combination(weights.iter().copied().zip(mapped.iter())).vector(),
        );
      }
    }
  }
}
