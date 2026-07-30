use std::marker::PhantomData;

use crate::{CoordSpace, Coords, CoordsRef, Matrix};

/// An affine map $x |-> A x + b$ from one coordinate space to another,
/// with linear part $A$ and translation $b$.
///
/// Equivalently, and this is what characterizes the affine maps among all maps:
/// it preserves affine combinations,
/// $T(sum_i lambda_i p_i) = sum_i lambda_i T(p_i)$ whenever $sum_i lambda_i = 1$
/// (see [`Coords::affine_combination`]).
///
/// The two spaces are type parameters, so a map carries its direction the way a
/// [`Coords`] carries its space: composition demands a shared middle space, the
/// wrong one does not compile, and [`Self::pseudo_inverse`] returns an
/// `AffineTransform<To, From>`, so a map and its inverse are told apart by their
/// types rather than by a naming convention.
///
/// The linear part need not be square. A tall injective $A$ is a map into a
/// higher-dimensional space, inverted on its image by [`Self::pseudo_inverse`].
///
/// $A$ stays a bare [`Matrix`]. It maps displacements, which this crate leaves
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
  /// The affine pseudo-inverse: the map $y |-> A^+ (y - b)$ inverting
  /// $x |-> A x + b$, and therefore a map the other way round.
  ///
  /// It sends $y$ to the least-squares preimage, the $x$ minimizing
  /// $norm(A x + b - y)$, so on an injective (full-column-rank) $A$ it is a
  /// genuine left inverse and on a bijective one the inverse. Total on the
  /// zero-dimensional domain.
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
