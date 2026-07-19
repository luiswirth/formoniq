use crate::{Matrix, Vector, VectorView};

/// An affine map $x |-> A x + b$ between coordinate spaces, with linear part
/// $A$ and translation $b$.
///
/// The linear part need not be square: a cell parametrization $hat(K) -> RR^N$
/// has a tall injective $A$, inverted in the least-squares sense by
/// [`Self::apply_backward`] and [`Self::pseudo_inverse`].
pub struct AffineTransform {
  pub translation: Vector,
  pub linear: Matrix,
}
impl AffineTransform {
  pub fn new(translation: Vector, linear: Matrix) -> Self {
    Self {
      translation,
      linear,
    }
  }
  pub fn dim_domain(&self) -> usize {
    self.linear.ncols()
  }
  pub fn dim_image(&self) -> usize {
    self.linear.nrows()
  }

  pub fn apply_forward(&self, coord: VectorView) -> Vector {
    &self.linear * coord + &self.translation
  }
  /// The least-squares preimage: the $x$ minimizing $norm(A x + b - y)$, from
  /// the SVD of $A$. On an injective $A$ it is the exact inverse of
  /// [`Self::apply_forward`]; total on the zero-dimensional domain.
  pub fn apply_backward(&self, coord: VectorView) -> Vector {
    if self.dim_domain() == 0 {
      return Vector::zeros(0);
    }
    self
      .linear
      .clone()
      .svd(true, true)
      .solve(&(coord - &self.translation), 1e-12)
      .unwrap()
  }

  /// The affine pseudo-inverse: the map $y |-> A^+ (y - b)$ inverting
  /// $x |-> A x + b$. On an injective (full-column-rank) $A$ it is a genuine
  /// left inverse, so its forward action agrees with [`Self::apply_backward`].
  pub fn pseudo_inverse(&self) -> Self {
    if self.dim_domain() == 0 {
      return Self::new(Vector::zeros(0), Matrix::zeros(0, self.dim_image()));
    }
    let linear = self.linear.clone().pseudo_inverse(1e-12).unwrap();
    let translation = -&linear * &self.translation;
    Self {
      translation,
      linear,
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn close(a: &Vector, b: &Vector) {
    assert_eq!(a.len(), b.len());
    assert!((a - b).norm() < 1e-9, "{a:?} != {b:?}");
  }

  // A deterministic full-column-rank `nrows x ncols` matrix (ncols <= nrows):
  // unit lower-triangular columns, echelon and hence injective.
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

  fn translation(dim: usize) -> Vector {
    Vector::from_fn(dim, |i, _| 1.0 + i as f64)
  }

  fn point(dim: usize) -> Vector {
    Vector::from_fn(dim, |i, _| 2.0 - 0.3 * i as f64)
  }

  /// On an injective map, `apply_backward` is a left inverse of `apply_forward`.
  #[test]
  fn backward_is_left_inverse_of_forward() {
    for image in 0..=4 {
      for domain in 0..=image {
        let t = AffineTransform::new(translation(image), full_col_rank(image, domain));
        let x = point(domain);
        let y = t.apply_forward(x.as_view());
        close(&t.apply_backward(y.as_view()), &x);
      }
    }
  }

  /// The pseudo-inverse's forward action is exactly `apply_backward`: the two
  /// spellings of $A^+(y - b)$ must agree, sign included.
  #[test]
  fn pseudo_inverse_forward_is_apply_backward() {
    for image in 0..=4 {
      for domain in 0..=image {
        let t = AffineTransform::new(translation(image), full_col_rank(image, domain));
        let inv = t.pseudo_inverse();
        let y = point(image);
        close(
          &inv.apply_forward(y.as_view()),
          &t.apply_backward(y.as_view()),
        );
      }
    }
  }

  /// For a square invertible map the pseudo-inverse is a two-sided inverse:
  /// `pinv ∘ forward = id` on the domain.
  #[test]
  fn pseudo_inverse_undoes_forward() {
    for dim in 0..=4 {
      let t = AffineTransform::new(translation(dim), full_col_rank(dim, dim));
      let inv = t.pseudo_inverse();
      let x = point(dim);
      let round = inv.apply_forward(t.apply_forward(x.as_view()).as_view());
      close(&round, &x);
    }
  }

  /// Degenerate domain: a map out of $RR^0$ is the constant `b`, its backward
  /// and pseudo-inverse land in the empty space. Total, no panic.
  #[test]
  fn zero_dimensional_domain_is_total() {
    let t = AffineTransform::new(translation(3), full_col_rank(3, 0));
    assert_eq!(t.apply_forward(Vector::zeros(0).as_view()), translation(3));
    assert_eq!(t.apply_backward(point(3).as_view()).len(), 0);
    let inv = t.pseudo_inverse();
    assert_eq!(inv.dim_domain(), 3);
    assert_eq!(inv.dim_image(), 0);
  }
}
