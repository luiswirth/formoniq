use crate::linalg::nalgebra::DMatrixExt;

pub type Dim = usize;

/// A Gram Matrix represent an inner product expressed in a basis.
#[derive(Debug, Clone)]
pub struct Gramian {
  /// S.P.D. matrix
  matrix: na::DMatrix<f64>,
}
impl Gramian {
  pub fn try_new(matrix: na::DMatrix<f64>) -> Option<Self> {
    matrix.is_spd().then_some(Self { matrix })
  }
  pub fn new(matrix: na::DMatrix<f64>) -> Self {
    Self::try_new(matrix).expect("Matrix must be s.p.d.")
  }
  pub fn new_unchecked(matrix: na::DMatrix<f64>) -> Self {
    if cfg!(debug_assertions) {
      Self::new(matrix)
    } else {
      Self { matrix }
    }
  }
  pub fn from_euclidean_vectors(vectors: na::DMatrix<f64>) -> Self {
    assert!(vectors.is_full_rank(1e-9), "Matrix must be full rank.");
    let matrix = vectors.transpose() * vectors;
    Self::new_unchecked(matrix)
  }
  /// Orthonormal euclidean metric.
  pub fn standard(dim: Dim) -> Self {
    let matrix = na::DMatrix::identity(dim, dim);
    Self::new_unchecked(matrix)
  }

  pub fn matrix(&self) -> &na::DMatrix<f64> {
    &self.matrix
  }
  pub fn dim(&self) -> Dim {
    self.matrix.nrows()
  }
  pub fn det(&self) -> f64 {
    self.matrix.determinant()
  }
  pub fn det_sqrt(&self) -> f64 {
    self.det().sqrt()
  }
  pub fn inverse(self) -> Self {
    let matrix = self
      .matrix
      .try_inverse()
      .expect("Symmetric Positive Definite is always invertible.");
    Self::new_unchecked(matrix)
  }
}

/// Inner product functionality directly on the basis.
impl Gramian {
  pub fn basis_inner(&self, i: usize, j: usize) -> f64 {
    self.matrix[(i, j)]
  }
  pub fn basis_norm_sq(&self, i: usize) -> f64 {
    self.basis_inner(i, i)
  }
  pub fn basis_norm(&self, i: usize) -> f64 {
    self.basis_norm_sq(i).sqrt()
  }
  pub fn basis_angle_cos(&self, i: usize, j: usize) -> f64 {
    self.basis_inner(i, j) / self.basis_norm(i) / self.basis_norm(j)
  }
  pub fn basis_angle(&self, i: usize, j: usize) -> f64 {
    self.basis_angle_cos(i, j).acos()
  }
}
impl std::ops::Index<(usize, usize)> for Gramian {
  type Output = f64;
  fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
    &self.matrix[(i, j)]
  }
}

/// Inner product functionality directly on any element.
impl Gramian {
  pub fn inner(&self, v: &na::DVector<f64>, w: &na::DVector<f64>) -> f64 {
    (v.transpose() * self.matrix() * w).x
  }
  pub fn inner_mat(&self, v: &na::DMatrix<f64>, w: &na::DMatrix<f64>) -> na::DMatrix<f64> {
    v.transpose() * self.matrix() * w
  }
  pub fn norm_sq(&self, v: &na::DVector<f64>) -> f64 {
    self.inner(v, v)
  }
  pub fn norm_sq_mat(&self, v: &na::DMatrix<f64>) -> na::DMatrix<f64> {
    self.inner_mat(v, v)
  }
  pub fn norm(&self, v: &na::DVector<f64>) -> f64 {
    self.inner(v, v).sqrt()
  }
  pub fn norm_mat(&self, v: &na::DMatrix<f64>) -> na::DMatrix<f64> {
    self.inner_mat(v, v).map(|v| v.sqrt())
  }
  pub fn angle_cos(&self, v: &na::DVector<f64>, w: &na::DVector<f64>) -> f64 {
    self.inner(v, w) / self.norm(v) / self.norm(w)
  }
  pub fn angle(&self, v: &na::DVector<f64>, w: &na::DVector<f64>) -> f64 {
    self.angle_cos(v, w).acos()
  }
}

impl Gramian {
  /// Orthonormal (w.r.t. metric) vectors expressed using old basis.
  ///
  /// Orthonormalizes metric tensor $I = B^T G B = B^T V^T V B = (V B)^T (V B)$
  pub fn orthonormal_basis(&self) -> na::DMatrix<f64> {
    let na::SymmetricEigen {
      eigenvalues,
      mut eigenvectors,
    } = self.matrix.clone().symmetric_eigen();
    for (eigenvalue, mut eigenvector) in eigenvalues.iter().zip(eigenvectors.column_iter_mut()) {
      eigenvector /= eigenvalue.sqrt();
    }
    eigenvectors
  }
  pub fn orthormal_change_of_basis(&self) -> na::DMatrix<f64> {
    self.orthonormal_basis().try_inverse().unwrap()
  }
}

#[cfg(test)]
mod test {
  use super::*;

  #[test]
  fn orthonormal_basis() {
    let dim = 2;
    let tangent_vectors = na::dmatrix![
      1.0, 0.0;
      1.0, 1.0;
    ];
    let metric = Gramian::from_euclidean_vectors(tangent_vectors.clone());
    let orthonormal_basis = metric.orthonormal_basis();
    assert!(
      (orthonormal_basis.transpose() * metric.matrix() * &orthonormal_basis
        - na::DMatrix::identity(dim, dim))
      .norm()
        <= 1e-12
    );

    let orthonormal_tangent_vectors = tangent_vectors * &orthonormal_basis;
    let orthogonal_metric = Gramian::from_euclidean_vectors(orthonormal_tangent_vectors);
    assert!((orthogonal_metric.matrix() - na::DMatrix::identity(dim, dim)).norm() <= 1e-12);

    let vector_a = na::dvector![0.2, 5.3];
    let vector_b = na::dvector![-1.3, 2.8];
    let inner0 = metric.inner(&vector_a, &vector_b);
    let cob = metric.orthormal_change_of_basis();
    let inner1 = (&cob * vector_a).dot(&(&cob * vector_b));
    assert_eq!(inner0, inner1);
  }
}
