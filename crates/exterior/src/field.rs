use common::{
  affine::AffineTransform,
  linalg::nalgebra::{Vector, VectorView},
};

use crate::{Dim, ExteriorGrade, MultiForm};

/// A differential form: a multiform field over a manifold.
pub trait ExteriorField {
  fn dim_ambient(&self) -> Dim;
  fn dim_intrinsic(&self) -> Dim;
  fn grade(&self) -> ExteriorGrade;
  fn at_point<'a>(&self, coord: impl Into<VectorView<'a>>) -> MultiForm;
}

pub type DiffFormClosure = ExteriorFieldClosure;

/// A pointwise evaluation rule for a differential form.
pub type FormClosure = Box<dyn Fn(VectorView<f64>) -> MultiForm + Sync>;

pub struct ExteriorFieldClosure {
  closure: FormClosure,
  dim: Dim,
  grade: ExteriorGrade,
}

impl ExteriorFieldClosure {
  pub fn new(closure: FormClosure, dim: Dim, grade: ExteriorGrade) -> Self {
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
    let wrapper = move |x: VectorView<f64>| MultiForm::scalar(f(x), dim);
    Self::new(Box::new(wrapper), dim, 0)
  }
  /// Create a 1-form (covector field).
  pub fn one_form(f: impl Fn(VectorView<f64>) -> Vector + Sync + 'static, dim: Dim) -> Self {
    let wrapper = move |x: VectorView<f64>| MultiForm::line(f(x));
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
  fn at_point<'a>(&self, coord: impl Into<VectorView<'a>>) -> MultiForm {
    (self.closure)(coord.into())
  }
}

/// The pullback of a differential form along an affine map.
pub struct FormPullback<F: ExteriorField> {
  form: F,
  affine_transform: AffineTransform,
}
impl<F: ExteriorField> FormPullback<F> {
  pub fn new(form: F, affine_transform: AffineTransform) -> Self {
    Self {
      form,
      affine_transform,
    }
  }
}
impl<F: ExteriorField> ExteriorField for FormPullback<F> {
  fn dim_ambient(&self) -> Dim {
    self.affine_transform.dim_image()
  }
  fn dim_intrinsic(&self) -> Dim {
    self.form.dim_intrinsic()
  }
  fn grade(&self) -> ExteriorGrade {
    self.form.grade()
  }
  fn at_point<'a>(&self, local: impl Into<VectorView<'a>>) -> MultiForm {
    let local = local.into();
    let global = self.affine_transform.apply_forward(local);
    let form_value = self.form.at_point(&global);
    form_value.pullback(&self.affine_transform.linear)
  }
}
