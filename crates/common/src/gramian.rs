use crate::{
  linalg::nalgebra::{Matrix, MatrixExt, Vector},
  Dim,
};

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
}
impl std::ops::Index<(usize, usize)> for Gramian {
  type Output = f64;
  fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
    &self.matrix[(i, j)]
  }
}

/// A Riemannian metric: the Gramian on tangent vectors together with its
/// inverse, the induced Gramian on covectors.
///
/// Keeping both eliminates the recurring question of whether a computation
/// needs $g$ or $g^(-1)$: contravariant quantities (vectors) are measured by
/// [`Self::vector_gramian`], covariant ones (forms) by
/// [`Self::covector_gramian`].
#[derive(Debug, Clone)]
pub struct RiemannianMetric {
  vector_gramian: Gramian,
  covector_gramian: Gramian,
}
impl RiemannianMetric {
  pub fn new(vector_gramian: Gramian) -> Self {
    let covector_gramian = vector_gramian.clone().inverse();
    Self {
      vector_gramian,
      covector_gramian,
    }
  }
  /// Orthonormal euclidean metric.
  pub fn standard(dim: Dim) -> Self {
    Self {
      vector_gramian: Gramian::standard(dim),
      covector_gramian: Gramian::standard(dim),
    }
  }

  pub fn dim(&self) -> Dim {
    self.vector_gramian.dim()
  }
  /// The metric tensor $g$: the inner product on tangent vectors.
  pub fn vector_gramian(&self) -> &Gramian {
    &self.vector_gramian
  }
  /// The inverse metric tensor $g^(-1)$: the inner product on covectors.
  pub fn covector_gramian(&self) -> &Gramian {
    &self.covector_gramian
  }
  /// $sqrt(det g)$: the volume scaling factor of the metric.
  pub fn det_sqrt(&self) -> f64 {
    self.vector_gramian.det_sqrt()
  }
}

/// Inner product functionality directly on any element.
impl Gramian {
  pub fn inner(&self, v: &Vector, w: &Vector) -> f64 {
    (v.transpose() * self.matrix() * w).x
  }
  pub fn norm_sq(&self, v: &Vector) -> f64 {
    self.inner(v, v)
  }
  pub fn norm(&self, v: &Vector) -> f64 {
    self.inner(v, v).sqrt()
  }
}
