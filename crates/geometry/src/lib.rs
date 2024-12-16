pub mod coord;
pub mod regge;

use common::Dim;

extern crate nalgebra as na;

#[derive(Debug, Clone)]
pub struct RiemannianMetric {
  metric_tensor: na::DMatrix<f64>,
  inverse_metric_tensor: na::DMatrix<f64>,
}
impl RiemannianMetric {
  pub fn new(metric_tensor: na::DMatrix<f64>) -> Self {
    // WARN: Numerically Unstable. TODO: can we avoid this?
    let inverse_metric_tensor = metric_tensor.clone().try_inverse().unwrap();
    Self {
      metric_tensor,
      inverse_metric_tensor,
    }
  }

  pub fn euclidean(dim: Dim) -> Self {
    let identity = na::DMatrix::identity(dim, dim);
    let metric_tensor = identity.clone();
    let inverse_metric_tensor = identity;
    Self {
      metric_tensor,
      inverse_metric_tensor,
    }
  }

  pub fn metric_tensor(&self) -> &na::DMatrix<f64> {
    &self.metric_tensor
  }
  pub fn inverse_metric_tensor(&self) -> &na::DMatrix<f64> {
    &self.inverse_metric_tensor
  }

  pub fn dim(&self) -> Dim {
    self.metric_tensor.nrows()
  }

  pub fn det(&self) -> f64 {
    self.metric_tensor.determinant()
  }
  pub fn det_sqrt(&self) -> f64 {
    self.det().sqrt()
  }

  /// Gram matrix on tangent vector standard basis.
  pub fn vector_gramian(&self) -> &na::DMatrix<f64> {
    &self.metric_tensor
  }
  pub fn vector_inner_product(
    &self,
    v: &na::DMatrix<f64>,
    w: &na::DMatrix<f64>,
  ) -> na::DMatrix<f64> {
    v.transpose() * self.vector_gramian() * w
  }
  pub fn vector_norm_sqr(&self, v: &na::DMatrix<f64>) -> na::DMatrix<f64> {
    self.vector_inner_product(v, v)
  }

  /// Gram matrix on tangent covector standard basis.
  pub fn covector_gramian(&self) -> &na::DMatrix<f64> {
    &self.inverse_metric_tensor
  }
  pub fn covector_inner_product(
    &self,
    v: &na::DMatrix<f64>,
    w: &na::DMatrix<f64>,
  ) -> na::DMatrix<f64> {
    v.transpose() * self.covector_gramian() * w
  }
  pub fn covector_norm_sqr(&self, v: &na::DMatrix<f64>) -> na::DMatrix<f64> {
    self.covector_inner_product(v, v)
  }
}
