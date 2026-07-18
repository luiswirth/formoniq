//! Coordinates, tagged by the space they live in.
//!
//! A vector of numbers is not a point: it is a point *in a coordinate system*,
//! and the same tuple means different things in different ones. Barycentric
//! weights $lambda in RR^(n+1)$, the cartesian coordinates $x in RR^n$ of a
//! chart, and the ambient coordinates of an embedding $RR^N$ are three distinct
//! spaces, and every map between them is a real map with a real inverse.
//!
//! [`Coords`] carries the space as a type parameter, so those maps have to be
//! written down and the wrong composition does not compile. The tag is a
//! zero-sized [`CoordSpace`] marker: no runtime cost, no runtime check. Reading
//! is unaffected -- `Coords` derefs to the underlying vector, so the whole
//! nalgebra API is there.
//!
//! The tag is a discipline, not a proof. Building a `Coords<S>` out of a raw
//! [`Vector`] -- through [`Coords::new`] or [`From`] -- is unchecked, and
//! deliberately so: that is the boundary at which untagged linear algebra
//! enters, and there is nothing at a boundary to check against. The space is
//! whatever the caller says it is. What the tags buy is that *past* the
//! boundary the claim is carried and enforced: a coordinate of one space cannot
//! be passed where another is expected, and a map between two spaces has to
//! exist and be named rather than be assumed.

pub mod affine;

extern crate nalgebra as na;

use std::marker::PhantomData;

/// The dimension of a space or object.
pub type Dim = usize;

pub type Vector<T = f64> = na::DVector<T>;
pub type Matrix<T = f64> = na::DMatrix<T>;
pub type VectorView<'a, T = f64> = na::DVectorView<'a, T>;

/// A flat coordinate space: the tag that distinguishes coordinate systems at
/// compile time. Uninhabited -- it is a name, never a value.
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

/// A point of the coordinate space `S`, as its coordinate tuple.
///
/// Derefs to the underlying [`Vector`], so all read-only linear algebra is
/// available directly. The difference of two points of the same space is a
/// displacement, and is therefore a bare [`Vector`], not a `Coords`.
pub struct Coords<S: CoordSpace> {
  entries: Vector,
  space: PhantomData<S>,
}

impl<S: CoordSpace> Coords<S> {
  pub fn new(entries: Vector) -> Self {
    Self {
      entries,
      space: PhantomData,
    }
  }
  pub fn zeros(dim: Dim) -> Self {
    Self::new(Vector::zeros(dim))
  }
  pub fn from_element(dim: Dim, value: f64) -> Self {
    Self::new(Vector::from_element(dim, value))
  }
  pub fn from_iterator(dim: Dim, iter: impl IntoIterator<Item = f64>) -> Self {
    Self::new(Vector::from_iterator(dim, iter))
  }

  /// The number of coordinates, which is the dimension of the space.
  pub fn dim(&self) -> Dim {
    self.entries.len()
  }

  /// The coordinate tuple, untagged: the escape hatch into raw linear algebra.
  pub fn vector(&self) -> &Vector {
    &self.entries
  }
  pub fn vector_mut(&mut self) -> &mut Vector {
    &mut self.entries
  }
  pub fn into_vector(self) -> Vector {
    self.entries
  }
  /// The coordinate tuple as an untagged view: `Copy`, and the form nalgebra's
  /// consuming combinators want.
  pub fn view(&self) -> VectorView<'_> {
    self.entries.as_view()
  }
  pub fn as_view(&self) -> CoordsRef<'_, S> {
    CoordsRef::new(self.entries.as_view())
  }
}

/// A borrowed point of the coordinate space `S`: a view, for coordinates stored
/// as the column of a matrix.
pub struct CoordsRef<'a, S: CoordSpace> {
  view: VectorView<'a>,
  space: PhantomData<S>,
}

impl<'a, S: CoordSpace> CoordsRef<'a, S> {
  pub fn new(view: VectorView<'a>) -> Self {
    Self {
      view,
      space: PhantomData,
    }
  }
  pub fn dim(&self) -> Dim {
    self.view.len()
  }
  pub fn view(&self) -> VectorView<'a> {
    self.view
  }
  pub fn to_coords(&self) -> Coords<S> {
    Coords::new(self.view.into_owned())
  }
}

impl<'a, S: CoordSpace> From<&'a Coords<S>> for CoordsRef<'a, S> {
  fn from(coords: &'a Coords<S>) -> Self {
    coords.as_view()
  }
}
/// The unchecked entry from raw linear algebra: the space is whatever the
/// context infers, claimed and not verified. The convenience is worth it, but
/// it is the one place the tagging is on trust (see the module docs).
impl<S: CoordSpace> From<Vector> for Coords<S> {
  fn from(entries: Vector) -> Self {
    Self::new(entries)
  }
}

impl<S: CoordSpace> std::ops::Deref for Coords<S> {
  type Target = Vector;
  fn deref(&self) -> &Self::Target {
    &self.entries
  }
}
impl<'a, S: CoordSpace> std::ops::Deref for CoordsRef<'a, S> {
  type Target = VectorView<'a>;
  fn deref(&self) -> &Self::Target {
    &self.view
  }
}

/// The displacement between two points of the same space: a tangent vector,
/// hence untagged.
impl<S: CoordSpace> std::ops::Sub for &Coords<S> {
  type Output = Vector;
  fn sub(self, rhs: Self) -> Vector {
    self.vector() - rhs.vector()
  }
}
impl<S: CoordSpace> std::ops::Sub<CoordsRef<'_, S>> for &Coords<S> {
  type Output = Vector;
  fn sub(self, rhs: CoordsRef<'_, S>) -> Vector {
    self.vector() - rhs.view()
  }
}
impl<S: CoordSpace> std::ops::Sub<&Coords<S>> for CoordsRef<'_, S> {
  type Output = Vector;
  fn sub(self, rhs: &Coords<S>) -> Vector {
    self.view() - rhs.vector()
  }
}
impl<S: CoordSpace> std::ops::Sub for CoordsRef<'_, S> {
  type Output = Vector;
  fn sub(self, rhs: Self) -> Vector {
    self.view() - rhs.view()
  }
}

// The derives would demand `S: Clone`, which a marker never is.
impl<S: CoordSpace> Clone for Coords<S> {
  fn clone(&self) -> Self {
    Self::new(self.entries.clone())
  }
}
impl<S: CoordSpace> Clone for CoordsRef<'_, S> {
  fn clone(&self) -> Self {
    *self
  }
}
impl<S: CoordSpace> Copy for CoordsRef<'_, S> {}

impl<S: CoordSpace> PartialEq for Coords<S> {
  fn eq(&self, other: &Self) -> bool {
    self.entries == other.entries
  }
}
impl<S: CoordSpace> PartialEq for CoordsRef<'_, S> {
  fn eq(&self, other: &Self) -> bool {
    self.view == other.view
  }
}
impl<S: CoordSpace> std::fmt::Debug for Coords<S> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "{}{:?}", S::NAME, self.entries.as_slice())
  }
}
impl<S: CoordSpace> std::fmt::Debug for CoordsRef<'_, S> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "{}{:?}", S::NAME, self.view.iter().collect::<Vec<_>>())
  }
}
