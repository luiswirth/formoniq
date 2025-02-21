use std::marker::PhantomData;

use common::metric::AffineDiffeomorphism;

use crate::{
  variance::{self, VarianceMarker},
  Dim, ExteriorElement, ExteriorGrade,
};

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

pub type DifferentialFormClosure = ExteriorFieldClosure<variance::Co>;

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
