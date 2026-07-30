#![doc = include_str!("../README.md")]

pub mod affine;

extern crate nalgebra as na;

use std::marker::PhantomData;

pub type Vector<T = f64> = na::DVector<T>;
pub type Matrix<T = f64> = na::DMatrix<T>;
pub type VectorView<'a, T = f64> = na::DVectorView<'a, T>;

/// A flat coordinate space: the tag that distinguishes coordinate systems at
/// compile time. Uninhabited: it is a name, never a value.
pub trait CoordSpace: 'static {
  /// The name of the space, for diagnostics.
  const NAME: &'static str;
}

/// The ambient space $RR^N$ of an embedding: where a mesh's vertex coordinates
/// live, and where mesh-independent analytic data is stated.
///
/// Extrinsic: it exists only when an embedding is given. An intrinsic manifold,
/// such as one presented by Regge edge lengths, has no ambient space at all.
pub enum Ambient {}
impl CoordSpace for Ambient {
  const NAME: &'static str = "ambient";
}

/// A point of the ambient space: where the vertices of an embedded mesh sit, and
/// the argument analytic coordinate data is a function of.
pub type Coord = Coords<Ambient>;
pub type CoordRef<'a> = CoordsRef<'a, Ambient>;

/// The coordinate tuple backing a point: owned, or a view into the column of a
/// matrix. What a point needs of its storage is that its coordinates can be
/// read.
pub trait CoordVector {
  fn view(&self) -> VectorView<'_>;
}
impl CoordVector for Vector {
  fn view(&self) -> VectorView<'_> {
    self.as_view()
  }
}
impl CoordVector for VectorView<'_> {
  fn view(&self) -> VectorView<'_> {
    self.as_view()
  }
}

/// A point of the affine space `S`, as its coordinate tuple.
///
/// Points and displacements are different things, and the type says which: the
/// difference of two points is a bare [`Vector`], a point plus a displacement is
/// a point again, and there is no sum of two points. That free transitive action
/// of $RR^n$ on the points is the affine structure, and
/// [`affine_combination`](Self::affine_combination) is the one way to combine
/// points without choosing an origin.
///
/// Derefs to the storage, so all read-only linear algebra is available directly.
pub struct Coords<S: CoordSpace, V: CoordVector = Vector> {
  entries: V,
  /// The tag is a label on the value, not data it holds, so it is phantom
  /// *output* rather than phantom ownership: what a point can be sent or
  /// shared across is decided by its coordinates alone, exactly as for the
  /// bare vector it tags, and never by the marker naming its space.
  space: PhantomData<fn() -> S>,
}

/// A borrowed point of the affine space `S`: a view, for coordinates stored as
/// the column of a matrix.
pub type CoordsRef<'a, S> = Coords<S, VectorView<'a>>;

impl<S: CoordSpace, V: CoordVector> Coords<S, V> {
  /// The tuple claims the space: this is where a raw coordinate vector enters,
  /// and the tag is asserted rather than derived. Every operation afterwards
  /// preserves it.
  pub fn new(entries: V) -> Self {
    Self {
      entries,
      space: PhantomData,
    }
  }

  /// The number of coordinates, which is the dimension of the space.
  pub fn dim(&self) -> usize {
    self.view().len()
  }

  /// The coordinate tuple, untagged: the escape hatch into raw linear algebra.
  pub fn view(&self) -> VectorView<'_> {
    self.entries.view()
  }
  /// The same point, borrowed.
  pub fn as_view(&self) -> CoordsRef<'_, S> {
    CoordsRef::new(self.view())
  }
  /// The same point, owned.
  pub fn to_coords(&self) -> Coords<S> {
    Coords::new(self.view().into_owned())
  }
}

impl<S: CoordSpace> Coords<S> {
  pub fn zeros(dim: usize) -> Self {
    Self::new(Vector::zeros(dim))
  }
  pub fn from_element(dim: usize, value: f64) -> Self {
    Self::new(Vector::from_element(dim, value))
  }
  pub fn from_iterator(dim: usize, iter: impl IntoIterator<Item = f64>) -> Self {
    Self::new(Vector::from_iterator(dim, iter))
  }

  /// The affine combination $sum_i lambda_i p_i$ of points weighted by
  /// $lambda$, which is well defined precisely because $sum_i lambda_i = 1$:
  /// it equals $p_0 + sum_i lambda_i (p_i - p_0)$ for any of the points as base,
  /// a point displaced by a vector. This is the structure that makes the space
  /// affine rather than linear, and the only combination of points there is.
  ///
  /// The weights must sum to one and the points must be at least one and of a
  /// common dimension.
  pub fn affine_combination<'a, C: Into<CoordsRef<'a, S>>>(
    weighted: impl IntoIterator<Item = (f64, C)>,
  ) -> Self {
    let mut weighted = weighted.into_iter();
    let (weight, point) = weighted
      .next()
      .expect("an affine combination is of at least one point");
    let mut total = weight;
    let mut entries = weight * point.into().view();
    for (weight, point) in weighted {
      total += weight;
      entries += weight * point.into().view();
    }
    debug_assert!(
      (total - 1.0).abs() < 1e-9,
      "the weights of an affine combination sum to one, got {total}"
    );
    Self::new(entries)
  }

  /// The barycenter of a finite set of points: their affine combination with
  /// equal weights, hence independent of any origin.
  pub fn barycenter<'a, C: Into<CoordsRef<'a, S>>>(points: impl IntoIterator<Item = C>) -> Self {
    let mut points = points.into_iter();
    let first = points
      .next()
      .expect("a barycenter is of at least one point")
      .into();
    let mut count = 1.0;
    let mut entries = first.view().into_owned();
    for point in points {
      count += 1.0;
      entries += point.into().view();
    }
    Self::new(entries / count)
  }

  pub fn vector(&self) -> &Vector {
    &self.entries
  }
  pub fn vector_mut(&mut self) -> &mut Vector {
    &mut self.entries
  }
  pub fn into_vector(self) -> Vector {
    self.entries
  }
}

impl<'a, S: CoordSpace> CoordsRef<'a, S> {
  /// The borrowed tuple, for as long as the point's own borrow lasts, which
  /// outlives the handle.
  pub fn into_view(self) -> VectorView<'a> {
    self.entries
  }
}

impl<'a, S: CoordSpace, V: CoordVector> From<&'a Coords<S, V>> for CoordsRef<'a, S> {
  fn from(coords: &'a Coords<S, V>) -> Self {
    coords.as_view()
  }
}
impl<S: CoordSpace, V: CoordVector> From<V> for Coords<S, V> {
  fn from(entries: V) -> Self {
    Self::new(entries)
  }
}

impl<S: CoordSpace, V: CoordVector> std::ops::Deref for Coords<S, V> {
  type Target = V;
  fn deref(&self) -> &Self::Target {
    &self.entries
  }
}

/// The displacement between two points of the same space: a tangent vector,
/// hence untagged.
impl<S: CoordSpace, V: CoordVector, W: CoordVector> std::ops::Sub<&Coords<S, W>> for &Coords<S, V> {
  type Output = Vector;
  fn sub(self, rhs: &Coords<S, W>) -> Vector {
    self.view() - rhs.view()
  }
}
impl<S: CoordSpace, V: CoordVector + Copy, W: CoordVector + Copy> std::ops::Sub<Coords<S, W>>
  for Coords<S, V>
{
  type Output = Vector;
  fn sub(self, rhs: Coords<S, W>) -> Vector {
    self.view() - rhs.view()
  }
}

/// A point displaced by a vector is a point: the action of $RR^n$ that makes the
/// space affine. Free and transitive, its orbit map being subtraction of points.
impl<S: CoordSpace, V: CoordVector> std::ops::Add<&Vector> for &Coords<S, V> {
  type Output = Coords<S>;
  fn add(self, rhs: &Vector) -> Coords<S> {
    Coords::new(self.view() + rhs)
  }
}
impl<S: CoordSpace, V: CoordVector> std::ops::Sub<&Vector> for &Coords<S, V> {
  type Output = Coords<S>;
  fn sub(self, rhs: &Vector) -> Coords<S> {
    Coords::new(self.view() - rhs)
  }
}
impl<S: CoordSpace> std::ops::AddAssign<&Vector> for Coords<S> {
  fn add_assign(&mut self, rhs: &Vector) {
    self.entries += rhs;
  }
}
impl<S: CoordSpace> std::ops::SubAssign<&Vector> for Coords<S> {
  fn sub_assign(&mut self, rhs: &Vector) {
    self.entries -= rhs;
  }
}

// The derives would demand `S: Clone`, which a marker never is.
impl<S: CoordSpace, V: CoordVector + Clone> Clone for Coords<S, V> {
  fn clone(&self) -> Self {
    Self::new(self.entries.clone())
  }
}
impl<S: CoordSpace, V: CoordVector + Copy> Copy for Coords<S, V> {}

impl<S: CoordSpace, V: CoordVector, W: CoordVector> PartialEq<Coords<S, W>> for Coords<S, V> {
  fn eq(&self, other: &Coords<S, W>) -> bool {
    self.view() == other.view()
  }
}
impl<S: CoordSpace, V: CoordVector> std::fmt::Debug for Coords<S, V> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "{}{:?}", S::NAME, self.view().iter().collect::<Vec<_>>())
  }
}
