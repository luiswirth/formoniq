use std::marker::PhantomData;

use crate::{CoordSpace, Coords, CoordsRef, Matrix};

/// An affine map $x |-> A x + b$ **from** one coordinate space **to** another,
/// with linear part $A$ and translation $b$.
///
/// The two spaces are type parameters, so a map carries its direction the way a
/// [`Coords`] carries its space, and the wrong composition does not compile.
/// This is what invariant 3 looks like on the *morphisms* rather than on the
/// points: a chart and a parametrization are inverse maps, and the difference
/// between them is exactly which of `From` and `To` is which.
///
/// [`Self::pseudo_inverse`] returns an `AffineTransform<To, From>`, so turning a
/// parametrization into a chart is visible in the signature rather than in a
/// naming convention.
///
/// The linear part need not be square: a cell parametrization $hat(K) -> RR^N$
/// has a tall injective $A$, inverted in the least-squares sense by
/// [`Self::apply_backward`] and [`Self::pseudo_inverse`].
///
/// $A$ stays a bare [`Matrix`]. It maps *displacements*, which this crate leaves
/// untagged on purpose (the difference of two points of one space is a tangent
/// vector, not a point), so there is nothing for a tag to say about it.
pub struct AffineTransform<From: CoordSpace, To: CoordSpace> {
  /// The image of the origin of `From`, hence a point of `To`.
  pub translation: Coords<To>,
  /// The differential $A$, constant because the map is affine.
  pub linear: Matrix,
  spaces: PhantomData<fn(From) -> To>,
}

impl<From: CoordSpace, To: CoordSpace> AffineTransform<From, To> {
  pub fn new(translation: Coords<To>, linear: Matrix) -> Self {
    assert_eq!(
      translation.dim(),
      linear.nrows(),
      "the translation is a point of the target space"
    );
    Self {
      translation,
      linear,
      spaces: PhantomData,
    }
  }
  pub fn dim_domain(&self) -> usize {
    self.linear.ncols()
  }
  pub fn dim_image(&self) -> usize {
    self.linear.nrows()
  }

  pub fn apply_forward(&self, coord: CoordsRef<'_, From>) -> Coords<To> {
    Coords::new(&self.linear * coord.view() + self.translation.vector())
  }
  /// The least-squares preimage: the $x$ minimizing $norm(A x + b - y)$, from
  /// the SVD of $A$. On an injective $A$ it is the exact inverse of
  /// [`Self::apply_forward`]; total on the zero-dimensional domain.
  pub fn apply_backward(&self, coord: CoordsRef<'_, To>) -> Coords<From> {
    if self.dim_domain() == 0 {
      return Coords::zeros(0);
    }
    Coords::new(
      self
        .linear
        .clone()
        .svd(true, true)
        .solve(&(coord.view() - self.translation.vector()), 1e-12)
        .unwrap(),
    )
  }

  /// The affine pseudo-inverse: the map $y |-> A^+ (y - b)$ inverting
  /// $x |-> A x + b$, and therefore a map the other way round. On an injective
  /// (full-column-rank) $A$ it is a genuine left inverse, so its forward action
  /// agrees with [`Self::apply_backward`].
  pub fn pseudo_inverse(&self) -> AffineTransform<To, From> {
    if self.dim_domain() == 0 {
      return AffineTransform::new(Coords::zeros(0), Matrix::zeros(0, self.dim_image()));
    }
    let linear = self.linear.clone().pseudo_inverse(1e-12).unwrap();
    let translation = Coords::new(-&linear * self.translation.vector());
    AffineTransform::new(translation, linear)
  }
}

impl<From: CoordSpace, Mid: CoordSpace> AffineTransform<From, Mid> {
  /// Composition, applying `self` first: $x |-> B (A x + a) + b$.
  ///
  /// The shared middle space is what makes the two composable, and it is the
  /// type parameter that says so. Associativity and the identity are the
  /// category laws a chart atlas is stated in.
  pub fn then<To: CoordSpace>(
    &self,
    outer: &AffineTransform<Mid, To>,
  ) -> AffineTransform<From, To> {
    AffineTransform::new(
      outer.apply_forward(self.translation.as_view()),
      &outer.linear * &self.linear,
    )
  }
}

impl<S: CoordSpace> AffineTransform<S, S> {
  /// The identity of the space of the given dimension: the unit of composition.
  pub fn identity(dim: usize) -> Self {
    Self::new(Coords::zeros(dim), Matrix::identity(dim, dim))
  }
}

// The derive would demand the space markers be `Clone`, which a marker never is.
impl<From: CoordSpace, To: CoordSpace> Clone for AffineTransform<From, To> {
  fn clone(&self) -> Self {
    Self::new(self.translation.clone(), self.linear.clone())
  }
}

impl<From: CoordSpace, To: CoordSpace> std::fmt::Debug for AffineTransform<From, To> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(
      f,
      "{} -> {} affine {}x{}",
      From::NAME,
      To::NAME,
      self.dim_image(),
      self.dim_domain()
    )
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{Ambient, Vector};

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

  fn translation<S: CoordSpace>(dim: usize) -> Coords<S> {
    Coords::new(Vector::from_fn(dim, |i, _| 1.0 + i as f64))
  }

  fn point<S: CoordSpace>(dim: usize) -> Coords<S> {
    Coords::new(Vector::from_fn(dim, |i, _| 2.0 - 0.3 * i as f64))
  }

  fn probe(image: usize, domain: usize) -> AffineTransform<Source, Target> {
    AffineTransform::new(translation(image), full_col_rank(image, domain))
  }

  /// On an injective map, `apply_backward` is a left inverse of `apply_forward`.
  #[test]
  fn backward_is_left_inverse_of_forward() {
    for image in 0..=4 {
      for domain in 0..=image {
        let t = probe(image, domain);
        let x: Coords<Source> = point(domain);
        let y = t.apply_forward(x.as_view());
        close(t.apply_backward(y.as_view()).vector(), x.vector());
      }
    }
  }

  /// The pseudo-inverse's forward action is exactly `apply_backward`: the two
  /// spellings of $A^+(y - b)$ must agree, sign included. Its type is the map
  /// the other way round, which is the whole content of inverting a chart.
  #[test]
  fn pseudo_inverse_forward_is_apply_backward() {
    for image in 0..=4 {
      for domain in 0..=image {
        let t = probe(image, domain);
        let inv: AffineTransform<Target, Source> = t.pseudo_inverse();
        let y: Coords<Target> = point(image);
        close(
          inv.apply_forward(y.as_view()).vector(),
          t.apply_backward(y.as_view()).vector(),
        );
      }
    }
  }

  /// For a square invertible map the pseudo-inverse is a two-sided inverse:
  /// composing the two gives the identity, in both orders.
  #[test]
  fn pseudo_inverse_undoes_forward() {
    for dim in 0..=4 {
      let t = probe(dim, dim);
      let inv = t.pseudo_inverse();
      let x: Coords<Source> = point(dim);
      let round = inv.apply_forward(t.apply_forward(x.as_view()).as_view());
      close(round.vector(), x.vector());

      let composed = t.then(&inv);
      let identity = AffineTransform::<Source, Source>::identity(dim);
      close(
        &(&composed.linear * x.vector()),
        &(&identity.linear * x.vector()),
      );
      close(composed.translation.vector(), identity.translation.vector());
    }
  }

  /// Composition is functorial and associative, and the identity is its unit:
  /// applying the composite is applying the parts in order. The shared middle
  /// space is what the type parameters enforce.
  #[test]
  fn composition_is_application_in_order() {
    for outer in 1..=4 {
      for mid in 1..=outer {
        for inner in 1..=mid {
          let first: AffineTransform<Source, Ambient> =
            AffineTransform::new(translation(mid), full_col_rank(mid, inner));
          let second: AffineTransform<Ambient, Target> =
            AffineTransform::new(translation(outer), full_col_rank(outer, mid));
          let x: Coords<Source> = point(inner);

          let composed = first.then(&second);
          close(
            composed.apply_forward(x.as_view()).vector(),
            second
              .apply_forward(first.apply_forward(x.as_view()).as_view())
              .vector(),
          );

          let with_identity = first.then(&AffineTransform::<Ambient, Ambient>::identity(mid));
          close(
            with_identity.apply_forward(x.as_view()).vector(),
            first.apply_forward(x.as_view()).vector(),
          );
        }
      }
    }
  }

  /// Degenerate domain: a map out of $RR^0$ is the constant `b`, its backward
  /// and pseudo-inverse land in the empty space. Total, no panic.
  #[test]
  fn zero_dimensional_domain_is_total() {
    let t = probe(3, 0);
    assert_eq!(
      t.apply_forward(Coords::zeros(0).as_view()).vector(),
      translation::<Target>(3).vector()
    );
    assert_eq!(t.apply_backward(point::<Target>(3).as_view()).dim(), 0);
    let inv = t.pseudo_inverse();
    assert_eq!(inv.dim_domain(), 3);
    assert_eq!(inv.dim_image(), 0);
  }
}
