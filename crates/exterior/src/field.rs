use crate::{
  variance::{self, VarianceMarker},
  Dim, ExteriorElement, ExteriorGrade,
};

use common::metric::AffineDiffeomorphism;

use std::marker::PhantomData;

pub trait ExteriorField {
  type Variance: VarianceMarker;
  fn dim(&self) -> Dim;
  fn grade(&self) -> ExteriorGrade;
  fn at_point<'a>(
    &self,
    coord: impl Into<na::DVectorView<'a, f64>>,
  ) -> ExteriorElement<Self::Variance>;
}

// Trait aliases.
pub trait MultiVectorField: ExteriorField<Variance = variance::Contra> {}
impl<T: ExteriorField<Variance = variance::Contra>> MultiVectorField for T {}
pub trait MultiFormField: ExteriorField<Variance = variance::Co> {}
impl<T: ExteriorField<Variance = variance::Co>> MultiFormField for T {}
pub trait DifferentialMultiForm: MultiFormField {}
impl<T: MultiFormField> DifferentialMultiForm for T {}

pub type DiffFormClosure = ExteriorFieldClosure<variance::Co>;

#[allow(clippy::type_complexity)]
pub struct ExteriorFieldClosure<V: VarianceMarker> {
  closure: Box<dyn Fn(na::DVectorView<f64>) -> ExteriorElement<V>>,
  dim: Dim,
  grade: ExteriorGrade,
  variance: PhantomData<V>,
}

#[allow(clippy::type_complexity)]
impl<V: VarianceMarker> ExteriorFieldClosure<V> {
  pub fn new(
    closure: Box<dyn Fn(na::DVectorView<f64>) -> ExteriorElement<V>>,
    dim: Dim,
    grade: ExteriorGrade,
  ) -> Self {
    Self {
      closure,
      dim,
      grade,
      variance: PhantomData,
    }
  }
}

// Convenience methods specifically for DifferentialFormClosure
impl DiffFormClosure {
  /// Create a scalar field (0-form).
  pub fn scalar(f: impl Fn(na::DVectorView<f64>) -> f64 + 'static, dim: Dim) -> Self {
    let wrapper = move |x: na::DVectorView<f64>| crate::ExteriorElement::scalar(f(x), dim);
    Self::new(Box::new(wrapper), dim, 0)
  }
  /// Create a 1-form (covector field).
  pub fn one_form(
    f: impl Fn(na::DVectorView<f64>) -> na::DVector<f64> + 'static,
    dim: Dim,
  ) -> Self {
    let wrapper = move |x: na::DVectorView<f64>| crate::ExteriorElement::line(f(x));
    Self::new(Box::new(wrapper), dim, 1)
  }

  /// Create a constant scalar field.
  pub fn constant_scalar(value: f64, dim: Dim) -> Self {
    Self::scalar(move |_| value, dim)
  }
  /// Create a scalar field that extracts a specific coordinate component.
  pub fn coordinate_component(icomp: usize, dim: Dim) -> Self {
    assert!(icomp < dim, "Component index out of bounds");
    Self::scalar(move |x| x[icomp], dim)
  }
  /// Create a scalar field of the radial distance from a center point.
  pub fn radial_scalar(center: na::DVector<f64>, dim: Dim) -> Self {
    Self::scalar(move |x| (&center - x).norm(), dim)
  }
}
impl<V: VarianceMarker> ExteriorField for ExteriorFieldClosure<V> {
  type Variance = V;
  fn dim(&self) -> Dim {
    self.dim
  }
  fn grade(&self) -> ExteriorGrade {
    self.grade
  }
  fn at_point<'a>(
    &self,
    coord: impl Into<na::DVectorView<'a, f64>>,
  ) -> ExteriorElement<Self::Variance> {
    (self.closure)(coord.into())
  }
}

pub struct FormPushforward<F: DifferentialMultiForm> {
  form: F,
  diffeomorphism: AffineDiffeomorphism,
}
impl<F: DifferentialMultiForm> FormPushforward<F> {
  pub fn new(form: F, diffeomorphism: AffineDiffeomorphism) -> Self {
    Self {
      form,
      diffeomorphism,
    }
  }
}
impl<F: DifferentialMultiForm> ExteriorField for FormPushforward<F> {
  type Variance = variance::Co;
  fn dim(&self) -> Dim {
    self.form.dim()
  }
  fn grade(&self) -> ExteriorGrade {
    self.form.grade()
  }
  fn at_point<'a>(
    &self,
    coord_global: impl Into<na::DVectorView<'a, f64>>,
  ) -> ExteriorElement<Self::Variance> {
    let coord_global = coord_global.into();
    let coord_ref = self.diffeomorphism.apply_backward(coord_global);
    let form_ref = self.form.at_point(&coord_ref);
    let linear_inv = self.diffeomorphism.linear_inv();
    form_ref.precompose(linear_inv)
  }
}
