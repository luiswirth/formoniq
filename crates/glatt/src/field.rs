//! Fields of exterior elements over a chart domain of the continuum.
//!
//! A [`CoordField`] is mesh-independent analytic data on the continuum $M$: an
//! exact solution, a source term, a boundary flux, given as a function of a
//! point of a coordinate domain $Omega subset RR^m$. It is not the
//! discrete-differential-form notion of a field, a section of the exterior
//! bundle over a simplicial manifold, which has no global coordinate to be a
//! function of. The two are connected by a variance-directed functor: covariant
//! coordinate fields pull back onto the mesh along the composite of the cell
//! parametrization and the continuum chart, contravariant ones are pushed
//! forward off it.
//!
//! The domain is the coordinate space `S`. Its default,
//! [`Ambient`], is the flat case $Omega = RR^N$ with
//! $phi = id$: analytic data stated directly in the ambient coordinates of an
//! embedding. A curvilinear chart of the continuum, spherical $(theta, phi)$
//! on $S^2$, polar on a disk, is a different `S`, but the domain is
//! deliberately not tagged by a per-parametrization marker: it derefs to a
//! bare vector, so a component read is untyped regardless, and the junction a
//! marker would guard (two curvilinear charts of one manifold) does not arise in
//! a manufactured-solution script. The type stays generic so a power user can
//! bring markers. The default pays nothing.

use coorder::{Ambient, CoordSpace, Coords, Vector};

use multialgebra::{Degree, Dim, ExteriorGrade, Tensor, Variance};

/// A field of exterior elements over a coordinate domain $Omega subset RR^m$,
/// of the given [`Variance`]: a differential form when covariant, a
/// multivector field when contravariant.
pub trait CoordField<S: CoordSpace = Ambient> {
  fn dim(&self) -> Dim;
  fn grade(&self) -> ExteriorGrade;
  fn at(&self, coord: &Coords<S>) -> Tensor;
}

/// A coordinate field given by a pointwise closure.
///
/// The closure is boxed so that fields of the same variance and grade share a
/// type: manufactured solutions come in heterogeneous families
/// $(omega, dif omega, Delta omega)$ that want to sit in one collection. The
/// dynamic call is one indirection per evaluation, off any hot inner loop, a
/// consumer that needs the speed monomorphizes over [`CoordField`] instead.
pub struct FieldClosure<S: CoordSpace = Ambient> {
  closure: Box<PointwiseFn<S>>,
  dim: Dim,
  grade: ExteriorGrade,
}

/// The pointwise law of a [`FieldClosure`]: a function of a domain coordinate.
type PointwiseFn<S> = dyn Fn(&Coords<S>) -> Tensor + Sync;

/// A differential form on a coordinate domain.
///
/// An alias, not a distinct type: the variance of the values a closure returns
/// is theirs to state, so it names the intent rather than constraining it.
pub type DiffFormClosure<S = Ambient> = FieldClosure<S>;

impl<S: CoordSpace> FieldClosure<S> {
  pub fn new(
    closure: impl Fn(&Coords<S>) -> Tensor + Sync + 'static,
    dim: impl Into<Dim>,
    grade: impl Into<ExteriorGrade>,
  ) -> Self {
    Self {
      closure: Box::new(closure),
      dim: dim.into(),
      grade: grade.into(),
    }
  }

  /// A scalar field: one slot at grade 0.
  pub fn scalar(f: impl Fn(&Coords<S>) -> f64 + Sync + 'static, dim: impl Into<Dim>) -> Self {
    let dim = dim.into();
    Self::new(
      move |x| Tensor::multiform(Vector::from_element(1, f(x)), dim, Degree::ZERO),
      dim,
      Degree::ZERO,
    )
  }
  /// A covector field $omega = sum_i omega_i dif x^i$, from its components in
  /// the standard dual basis: the covariant grade-1 field, of which
  /// [`Self::vector_field`] is the contravariant one.
  pub fn one_form(f: impl Fn(&Coords<S>) -> Vector + Sync + 'static, dim: impl Into<Dim>) -> Self {
    Self::new(
      move |x| Tensor::line(f(x), Variance::Covariant),
      dim,
      Degree::ONE,
    )
  }
  /// A vector field $v = sum_i v^i partial_i$: the contravariant grade-1 field, of
  /// which [`Self::one_form`] is the covariant one.
  pub fn vector_field(
    f: impl Fn(&Coords<S>) -> Vector + Sync + 'static,
    dim: impl Into<Dim>,
  ) -> Self {
    Self::new(
      move |x| Tensor::line(f(x), Variance::Contravariant),
      dim,
      Degree::ONE,
    )
  }

  pub fn constant_scalar(value: f64, dim: impl Into<Dim>) -> Self {
    Self::scalar(move |_| value, dim)
  }
  /// The scalar field extracting one coordinate component, $x |-> x_i$.
  pub fn coord_component(icomp: usize, dim: impl Into<Dim>) -> Self {
    let dim = dim.into();
    assert!(icomp < dim, "Component index out of bounds");
    Self::scalar(move |x| x[icomp], dim)
  }
  /// The scalar field of the radial distance from a center point: the norm of
  /// the displacement between two points of the domain.
  pub fn radial_scalar(center: Coords<S>, dim: impl Into<Dim>) -> Self {
    Self::scalar(move |x| (x - &center).norm(), dim)
  }
}

impl<S: CoordSpace> CoordField<S> for FieldClosure<S> {
  fn dim(&self) -> Dim {
    self.dim
  }
  fn grade(&self) -> ExteriorGrade {
    self.grade
  }
  fn at(&self, coord: &Coords<S>) -> Tensor {
    (self.closure)(coord)
  }
}
