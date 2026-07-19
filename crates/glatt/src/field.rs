//! Fields of exterior elements over a chart domain of the continuum.
//!
//! A [`CoordField`] is mesh-independent analytic data on the continuum $M$: an
//! exact solution, a source term, a boundary flux, given as a function of a
//! point of a coordinate domain $Omega subset RR^m$. It is *not* the
//! discrete-differential-form notion of a field -- a section of the exterior
//! bundle over a simplicial manifold, which has no global coordinate to be a
//! function of. The two are connected by a variance-directed functor: covariant
//! coordinate fields pull back onto the mesh along the composite of the cell
//! parametrization and the continuum chart, contravariant ones are pushed
//! forward off it.
//!
//! The domain is the coordinate space `S`. Its default,
//! [`Ambient`], is the flat case $Omega = RR^N$ with
//! $phi = id$: analytic data stated directly in the ambient coordinates of an
//! embedding. A curvilinear chart of the continuum -- spherical $(theta, phi)$
//! on $S^2$, polar on a disk -- is a different `S`, but the domain is
//! deliberately *not* tagged by a per-parametrization marker: it derefs to a
//! bare vector, so a component read is untyped regardless, and the junction a
//! marker would guard (two curvilinear charts of one manifold) does not arise in
//! a manufactured-solution script. The type stays generic so a power user *can*
//! bring markers; the default pays nothing.

use coorder::{Ambient, CoordSpace, Coords, Vector};

use exterior::{Contravariant, Covariant, Dim, ExteriorElement, ExteriorGrade, Variance};

/// A field of exterior elements over a coordinate domain $Omega subset RR^m$,
/// of the given [`Variance`]: a differential form when covariant, a
/// multivector field when contravariant.
pub trait CoordField<V: Variance, S: CoordSpace = Ambient> {
  fn dim(&self) -> Dim;
  fn grade(&self) -> ExteriorGrade;
  fn at(&self, coord: &Coords<S>) -> ExteriorElement<V>;
}

/// A coordinate field given by a pointwise closure.
///
/// The closure is boxed so that fields of the same variance and grade share a
/// type: manufactured solutions come in heterogeneous families
/// $(omega, dif omega, Delta omega)$ that want to sit in one collection. The
/// dynamic call is one indirection per evaluation, off any hot inner loop -- a
/// consumer that needs the speed monomorphizes over [`CoordField`] instead.
pub struct FieldClosure<V: Variance, S: CoordSpace = Ambient> {
  closure: Box<PointwiseFn<V, S>>,
  dim: Dim,
  grade: ExteriorGrade,
}

/// The pointwise law of a [`FieldClosure`]: a function of a domain coordinate.
type PointwiseFn<V, S> = dyn Fn(&Coords<S>) -> ExteriorElement<V> + Sync;

/// A differential form on a coordinate domain: a covariant [`FieldClosure`].
pub type DiffFormClosure<S = Ambient> = FieldClosure<Covariant, S>;
/// A multivector field on a coordinate domain: a contravariant
/// [`FieldClosure`].
pub type MultiVectorFieldClosure<S = Ambient> = FieldClosure<Contravariant, S>;

impl<V: Variance, S: CoordSpace> FieldClosure<V, S> {
  pub fn new(
    closure: impl Fn(&Coords<S>) -> ExteriorElement<V> + Sync + 'static,
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
  pub fn scalar(f: impl Fn(&Coords<S>) -> f64 + Sync + 'static, dim: Dim) -> Self {
    Self::new(move |x| ExteriorElement::scalar(f(x), dim), dim, 0)
  }
  /// A grade-1 field, from its coefficients in the standard basis.
  pub fn line(f: impl Fn(&Coords<S>) -> Vector + Sync + 'static, dim: Dim) -> Self {
    Self::new(move |x| ExteriorElement::line(f(x)), dim, 1)
  }

  pub fn constant_scalar(value: f64, dim: Dim) -> Self {
    Self::scalar(move |_| value, dim)
  }
  /// The scalar field extracting one coordinate component, $x |-> x_i$.
  pub fn coord_component(icomp: usize, dim: Dim) -> Self {
    assert!(icomp < dim, "Component index out of bounds");
    Self::scalar(move |x| x[icomp], dim)
  }
  /// The scalar field of the radial distance from a center point.
  pub fn radial_scalar(center: Coords<S>, dim: Dim) -> Self {
    let center = center.into_vector();
    Self::scalar(move |x| (&center - x.vector()).norm(), dim)
  }
}

impl<S: CoordSpace> DiffFormClosure<S> {
  /// A covector field $omega = sum_i omega_i dif x^i$.
  pub fn one_form(f: impl Fn(&Coords<S>) -> Vector + Sync + 'static, dim: Dim) -> Self {
    Self::line(f, dim)
  }
}
impl<S: CoordSpace> MultiVectorFieldClosure<S> {
  /// A vector field $v = sum_i v^i diff_i$.
  pub fn vector_field(f: impl Fn(&Coords<S>) -> Vector + Sync + 'static, dim: Dim) -> Self {
    Self::line(f, dim)
  }
}

impl<V: Variance, S: CoordSpace> CoordField<V, S> for FieldClosure<V, S> {
  fn dim(&self) -> Dim {
    self.dim
  }
  fn grade(&self) -> ExteriorGrade {
    self.grade
  }
  fn at(&self, coord: &Coords<S>) -> ExteriorElement<V> {
    (self.closure)(coord)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use approx::assert_relative_eq;
  use coorder::Coord;

  // The pullback naturality law for `CoordField` — the field pulled onto a mesh
  // through a cell parametrization — lives in `derham`, the crate that joins
  // `glatt` to `simplicial` (see `derham::section`). What is testable here is
  // that each constructor evaluates to the exterior element it promises: the
  // pointwise law from which that naturality is built.

  fn point(entries: &[f64]) -> Coord {
    Coords::new(Vector::from_column_slice(entries))
  }

  /// A scalar field is grade 0, and evaluates to its closure's value.
  #[test]
  fn scalar_evaluates_pointwise() {
    for dim in 0..=4 {
      let f = DiffFormClosure::scalar(|x| x.iter().sum(), dim);
      assert_eq!(f.grade(), 0);
      assert_eq!(f.dim(), dim);
      let x = point(&vec![1.5; dim]);
      let value = f.at(&x);
      assert_eq!(value.grade(), 0);
      assert_relative_eq!(value.coeffs()[0], 1.5 * dim as f64);
    }
  }

  /// `coord_component` extracts $x_i$, and `constant_scalar` ignores its
  /// argument.
  #[test]
  fn distinguished_scalar_fields() {
    let dim = 3;
    let x = point(&[2.0, -1.0, 4.0]);
    for i in 0..dim {
      let f = DiffFormClosure::coord_component(i, dim);
      assert_relative_eq!(f.at(&x).coeffs()[0], x[i]);
    }
    let c = DiffFormClosure::constant_scalar(7.0, dim);
    assert_relative_eq!(c.at(&x).coeffs()[0], 7.0);
  }

  /// The radial field is the Euclidean distance to its center.
  #[test]
  fn radial_scalar_is_distance() {
    let dim = 2;
    let center = point(&[1.0, 1.0]);
    let f = DiffFormClosure::radial_scalar(center.clone(), dim);
    let x = point(&[4.0, 5.0]);
    assert_relative_eq!(f.at(&x).coeffs()[0], 5.0); // 3-4-5 triangle
    assert_relative_eq!(f.at(&center).coeffs()[0], 0.0);
  }

  /// A grade-1 field carries its coefficient vector verbatim, at either
  /// variance: `one_form` as a covector, `vector_field` as a vector.
  #[test]
  fn grade_one_fields_carry_their_coefficients() {
    let dim = 3;
    let x = point(&[0.0, 0.0, 0.0]);
    let coeffs = Vector::from_column_slice(&[1.0, 2.0, 3.0]);

    let omega = DiffFormClosure::one_form(
      {
        let c = coeffs.clone();
        move |_| c.clone()
      },
      dim,
    );
    assert_eq!(omega.grade(), 1);
    assert_eq!(omega.at(&x).coeffs(), &coeffs);

    let v = MultiVectorFieldClosure::vector_field(
      {
        let c = coeffs.clone();
        move |_| c.clone()
      },
      dim,
    );
    assert_eq!(v.grade(), 1);
    assert_eq!(v.at(&x).coeffs(), &coeffs);
  }
}
