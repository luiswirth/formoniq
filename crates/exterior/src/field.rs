use common::{
  affine::AffineTransform,
  linalg::nalgebra::{Vector, VectorView},
};

use crate::{Dim, ExteriorElement, ExteriorGrade};

pub trait ExteriorField {
  fn dim_ambient(&self) -> Dim;
  fn dim_intrinsic(&self) -> Dim;
  fn grade(&self) -> ExteriorGrade;
  fn at_point<'a>(&self, coord: impl Into<VectorView<'a>>) -> ExteriorElement;
}

// Trait aliases.
pub trait MultiVectorField: ExteriorField {}
impl<T: ExteriorField> MultiVectorField for T {}
pub trait MultiFormField: ExteriorField {}
impl<T: ExteriorField> MultiFormField for T {}
pub trait DifferentialMultiForm: MultiFormField {}
impl<T: MultiFormField> DifferentialMultiForm for T {}

pub type DiffFormClosure = ExteriorFieldClosure;

#[allow(clippy::type_complexity)]
pub struct ExteriorFieldClosure {
  closure: Box<dyn Fn(VectorView<f64>) -> ExteriorElement + Sync>,
  dim: Dim,
  grade: ExteriorGrade,
}

#[allow(clippy::type_complexity)]
impl ExteriorFieldClosure {
  pub fn new(
    closure: Box<dyn Fn(VectorView<f64>) -> ExteriorElement + Sync>,
    dim: Dim,
    grade: ExteriorGrade,
  ) -> Self {
    Self {
      closure,
      dim,
      grade,
    }
  }
}

// Convenience methods specifically for DifferentialFormClosure
impl DiffFormClosure {
  /// Create a scalar field (0-form).
  pub fn scalar(f: impl Fn(VectorView<f64>) -> f64 + Sync + 'static, dim: Dim) -> Self {
    let wrapper = move |x: VectorView<f64>| crate::ExteriorElement::scalar(f(x), dim);
    Self::new(Box::new(wrapper), dim, 0)
  }
  /// Create a 1-form (covector field).
  pub fn one_form(f: impl Fn(VectorView<f64>) -> Vector + Sync + 'static, dim: Dim) -> Self {
    let wrapper = move |x: VectorView<f64>| crate::ExteriorElement::line(f(x));
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
  pub fn radial_scalar(center: Vector, dim: Dim) -> Self {
    Self::scalar(move |x| (&center - x).norm(), dim)
  }
}
impl ExteriorField for ExteriorFieldClosure {
  fn dim_ambient(&self) -> Dim {
    self.dim
  }
  fn dim_intrinsic(&self) -> Dim {
    self.dim
  }
  fn grade(&self) -> ExteriorGrade {
    self.grade
  }
  fn at_point<'a>(&self, coord: impl Into<VectorView<'a>>) -> ExteriorElement {
    (self.closure)(coord.into())
  }
}

pub struct FormPullback<F: DifferentialMultiForm> {
  form: F,
  affine_transform: AffineTransform,
}
impl<F: DifferentialMultiForm> FormPullback<F> {
  pub fn new(form: F, affine_transform: AffineTransform) -> Self {
    Self {
      form,
      affine_transform,
    }
  }
}
impl<F: DifferentialMultiForm> ExteriorField for FormPullback<F> {
  fn dim_ambient(&self) -> Dim {
    self.affine_transform.dim_image()
  }
  fn dim_intrinsic(&self) -> Dim {
    self.form.dim_intrinsic()
  }
  fn grade(&self) -> ExteriorGrade {
    self.form.grade()
  }
  fn at_point<'a>(&self, local: impl Into<VectorView<'a>>) -> ExteriorElement {
    let local = local.into();
    let global = self.affine_transform.apply_forward(local);
    let form_ref = self.form.at_point(&global);
    let pushforward = &self.affine_transform.linear;
    form_ref.precompose_form(pushforward)
  }
}
