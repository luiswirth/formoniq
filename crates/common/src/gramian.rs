use crate::linalg::nalgebra::{Matrix, MatrixExt, Vector};

pub type Dim = usize;

/// A Gram Matrix represent an inner product expressed in a basis.
#[derive(Debug, Clone)]
pub struct Gramian {
  /// S.P.D. matrix
  matrix: Matrix,
}
impl Gramian {
  pub fn new(matrix: Matrix) -> Self {
    assert!(matrix.is_spd(), "Matrix must be s.p.d.");
    Self { matrix }
  }
  pub fn new_unchecked(matrix: Matrix) -> Self {
    if cfg!(debug_assertions) {
      Self::new(matrix)
    } else {
      Self { matrix }
    }
  }
  pub fn from_euclidean_vectors(vectors: Matrix) -> Self {
    assert!(vectors.is_full_rank(1e-9), "Matrix must be full rank.");
    let matrix = vectors.transpose() * vectors;
    Self::new_unchecked(matrix)
  }
  /// Orthonormal euclidean metric.
  pub fn standard(dim: Dim) -> Self {
    let matrix = Matrix::identity(dim, dim);
    Self::new_unchecked(matrix)
  }

  pub fn matrix(&self) -> &Matrix {
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
  pub fn inner(&self, v: &Vector, w: &Vector) -> f64 {
    (v.transpose() * self.matrix() * w).x
  }
  pub fn inner_mat(&self, v: &Matrix, w: &Matrix) -> Matrix {
    v.transpose() * self.matrix() * w
  }
  pub fn norm_sq(&self, v: &Vector) -> f64 {
    self.inner(v, v)
  }
  pub fn norm_sq_mat(&self, v: &Matrix) -> Matrix {
    self.inner_mat(v, v)
  }
  pub fn norm(&self, v: &Vector) -> f64 {
    self.inner(v, v).sqrt()
  }
  pub fn norm_mat(&self, v: &Matrix) -> Matrix {
    self.inner_mat(v, v).map(|v| v.sqrt())
  }
  pub fn angle_cos(&self, v: &Vector, w: &Vector) -> f64 {
    self.inner(v, w) / self.norm(v) / self.norm(w)
  }
  pub fn angle(&self, v: &Vector, w: &Vector) -> f64 {
    self.angle_cos(v, w).acos()
  }
}
