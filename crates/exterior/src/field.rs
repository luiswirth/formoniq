//! Fields of exterior elements on a flat coordinate domain.
//!
//! A [`CoordField`] is mesh-independent analytic data: an exact solution, a
//! source term, a boundary flux, given as a function of a point of
//! $Omega subset RR^n$. It is *not* the discrete-differential-form notion of a
//! field -- a section of the exterior bundle over a manifold, which lives one
//! crate up in `ddf` and has no global coordinate to be a function of. The two
//! are connected by the variance-directed functor: covariant coordinate fields
//! pull back onto the manifold along the chart, contravariant ones are pushed
//! forward off it.

use common::linalg::nalgebra::{Vector, VectorView};

use crate::{Contravariant, Covariant, Dim, ExteriorElement, ExteriorGrade, Variance};

/// A field of exterior elements over a flat coordinate domain in $RR^n$,
/// of the given [`Variance`]: a differential form when covariant, a
/// multivector field when contravariant.
pub trait CoordField<V: Variance> {
  fn dim(&self) -> Dim;
  fn grade(&self) -> ExteriorGrade;
  fn at<'a>(&self, coord: impl Into<VectorView<'a>>) -> ExteriorElement<V>;
}

/// A coordinate field given by a pointwise closure.
///
/// The closure is boxed so that fields of the same variance and grade share a
/// type: manufactured solutions come in heterogeneous families
/// $(omega, dif omega, Delta omega)$ that want to sit in one collection. The
/// dynamic call is one indirection per quadrature point, off the assembly hot
/// path -- the element loops themselves stay monomorphized over
/// [`CoordField`].
pub struct FieldClosure<V: Variance> {
  closure: Box<dyn Fn(VectorView) -> ExteriorElement<V> + Sync>,
  dim: Dim,
  grade: ExteriorGrade,
}

/// A differential form on a coordinate domain: a covariant [`FieldClosure`].
pub type DiffFormClosure = FieldClosure<Covariant>;
/// A multivector field on a coordinate domain: a contravariant
/// [`FieldClosure`].
pub type MultiVectorFieldClosure = FieldClosure<Contravariant>;

impl<V: Variance> FieldClosure<V> {
  pub fn new(
    closure: impl Fn(VectorView) -> ExteriorElement<V> + Sync + 'static,
    dim: Dim,
    grade: ExteriorGrade,
  ) -> Self {
    Self {
      closure: Box::new(closure),
      dim,
      grade,
    }
  }

  /// A scalar field: grade 0, where the two variances coincide.
  pub fn scalar(f: impl Fn(VectorView) -> f64 + Sync + 'static, dim: Dim) -> Self {
    Self::new(move |x| ExteriorElement::scalar(f(x), dim), dim, 0)
  }
  /// A grade-1 field, from its coefficients in the standard basis.
  pub fn line(f: impl Fn(VectorView) -> Vector + Sync + 'static, dim: Dim) -> Self {
    Self::new(move |x| ExteriorElement::line(f(x)), dim, 1)
  }

  pub fn constant_scalar(value: f64, dim: Dim) -> Self {
    Self::scalar(move |_| value, dim)
  }
  /// The scalar field extracting one coordinate component, $x |-> x_i$.
  pub fn coordinate_component(icomp: usize, dim: Dim) -> Self {
    assert!(icomp < dim, "Component index out of bounds");
    Self::scalar(move |x| x[icomp], dim)
  }
  /// The scalar field of the radial distance from a center point.
  pub fn radial_scalar(center: Vector, dim: Dim) -> Self {
    Self::scalar(move |x| (&center - x).norm(), dim)
  }
}

impl DiffFormClosure {
  /// A covector field $omega = sum_i omega_i dif x^i$.
  pub fn one_form(f: impl Fn(VectorView) -> Vector + Sync + 'static, dim: Dim) -> Self {
    Self::line(f, dim)
  }
}
impl MultiVectorFieldClosure {
  /// A vector field $v = sum_i v^i diff_i$.
  pub fn vector_field(f: impl Fn(VectorView) -> Vector + Sync + 'static, dim: Dim) -> Self {
    Self::line(f, dim)
  }
}

impl<V: Variance> CoordField<V> for FieldClosure<V> {
  fn dim(&self) -> Dim {
    self.dim
  }
  fn grade(&self) -> ExteriorGrade {
    self.grade
  }
  fn at<'a>(&self, coord: impl Into<VectorView<'a>>) -> ExteriorElement<V> {
    (self.closure)(coord.into())
  }
}
