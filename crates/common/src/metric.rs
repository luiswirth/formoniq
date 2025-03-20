use crate::linalg::DMatrixExt;

pub type Dim = usize;

#[derive(Debug, Clone)]
pub struct RiemannianMetric {
  metric_tensor: na::DMatrix<f64>,
  inverse_metric_tensor: na::DMatrix<f64>,
}
impl RiemannianMetric {
  pub fn new(metric_tensor: na::DMatrix<f64>) -> Self {
    let n = metric_tensor.nrows();
    // WARN: Numerically Unstable. TODO: can we avoid this?
    let inverse_metric_tensor = metric_tensor
      .clone()
      .cholesky()
      .unwrap()
      .solve(&na::DMatrix::identity(n, n));
    Self {
      metric_tensor,
      inverse_metric_tensor,
    }
  }

  pub fn from_tangent_basis(basis: na::DMatrix<f64>) -> Self {
    let metric_tensor = basis.gramian();
    Self::new(metric_tensor)
  }

  /// Orthonormal euclidean metric.
  pub fn standard(dim: Dim) -> Self {
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

  pub fn inner(&self, i: usize, j: usize) -> f64 {
    self.metric_tensor[(i, j)]
  }
  pub fn length_sqr(&self, i: usize) -> f64 {
    self.inner(i, i)
  }
  pub fn length(&self, i: usize) -> f64 {
    self.length_sqr(i).sqrt()
  }
  pub fn angle_cos(&self, i: usize, j: usize) -> f64 {
    self.inner(i, j) / self.length(i) / self.length(j)
  }
  pub fn angle(&self, i: usize, j: usize) -> f64 {
    self.angle_cos(i, j).acos()
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

  /// Orthonormal (w.r.t. metric) vectors expressed using old basis.
  ///
  /// Orthonormalizes metric tensor $I = B^T G B = B^T V^T V B = (V B)^T (V B)$
  pub fn orthonormal_basis(&self) -> na::DMatrix<f64> {
    let na::SymmetricEigen {
      eigenvalues,
      mut eigenvectors,
    } = self.metric_tensor.clone().symmetric_eigen();
    for (eigenvalue, mut eigenvector) in eigenvalues.iter().zip(eigenvectors.column_iter_mut()) {
      eigenvector /= eigenvalue.sqrt();
    }
    eigenvectors
  }

  pub fn orthormal_change_of_basis(&self) -> na::DMatrix<f64> {
    self.orthonormal_basis().try_inverse().unwrap()
  }

  /// Gram matrix on tangent vector standard basis.
  pub fn vector_gramian(&self) -> &na::DMatrix<f64> {
    &self.metric_tensor
  }

  pub fn vector_inner_product(&self, v: &na::DVector<f64>, w: &na::DVector<f64>) -> f64 {
    (v.transpose() * self.vector_gramian() * w).x
  }

  pub fn vector_inner_product_mat(
    &self,
    v: &na::DMatrix<f64>,
    w: &na::DMatrix<f64>,
  ) -> na::DMatrix<f64> {
    v.transpose() * self.vector_gramian() * w
  }
  pub fn vector_norm_sqr(&self, v: &na::DMatrix<f64>) -> na::DMatrix<f64> {
    self.vector_inner_product_mat(v, v)
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

pub struct AffineTransform {
  pub translation: na::DVector<f64>,
  pub linear: na::DMatrix<f64>,
}
impl AffineTransform {
  pub fn new(translation: na::DVector<f64>, linear: na::DMatrix<f64>) -> Self {
    Self {
      translation,
      linear,
    }
  }

  pub fn apply_forward(&self, coord: na::DVectorView<f64>) -> na::DVector<f64> {
    &self.linear * coord + &self.translation
  }
  pub fn apply_backward(&self, coord: na::DVectorView<f64>) -> na::DVector<f64> {
    if self.linear.is_empty() {
      return na::DVector::default();
    }
    self
      .linear
      .clone()
      .svd(true, true)
      .solve(&(coord - &self.translation), 1e-12)
      .unwrap()
  }

  pub fn pseudo_inverse(&self) -> Self {
    let linear = self.linear.clone().pseudo_inverse(1e-12).unwrap();
    let translation = &linear * &self.translation;
    Self {
      translation,
      linear,
    }
  }
}
