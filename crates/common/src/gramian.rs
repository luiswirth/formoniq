use crate::{
  linalg::nalgebra::{Matrix, MatrixExt, Vector},
  Dim,
};

/// A Gram Matrix represent an inner product expressed in a basis.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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

/// Inner product functionality expressed directly in terms of the basis.
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
  /// Cosine of the angle between basis vectors `i` and `j`.
  pub fn basis_angle_cos(&self, i: usize, j: usize) -> f64 {
    self.basis_inner(i, j) / self.basis_norm(i) / self.basis_norm(j)
  }
  /// Angle (in radians) between basis vectors `i` and `j`.
  pub fn basis_angle(&self, i: usize, j: usize) -> f64 {
    self.basis_angle_cos(i, j).acos()
  }

  /// Squared distance between two of the $"dim"+1$ points this Gramian is the
  /// edge metric of: vertex $0$ (the Gramian's own origin) and its $"dim"$
  /// basis vectors, read as vertices $1..="dim"$. $d(0,j)^2$ is the basis
  /// vector's own norm; $d(i,j)^2$ for $i,j >= 1$ follows from the law of
  /// cosines applied to the two edges from the origin,
  /// $d(i,j)^2 = g_(i-1,i-1) + g_(j-1,j-1) - 2 g_(i-1,j-1)$.
  fn vertex_dist_sq(&self, i: usize, j: usize) -> f64 {
    if i == j {
      0.0
    } else if i == 0 {
      self.basis_norm_sq(j - 1)
    } else if j == 0 {
      self.basis_norm_sq(i - 1)
    } else {
      self.basis_norm_sq(i - 1) + self.basis_norm_sq(j - 1) - 2.0 * self.basis_inner(i - 1, j - 1)
    }
  }

  /// The interior angle at vertex `v`, between its edges to vertices `a` and
  /// `b` (all in $0..="dim"$, vertex $0$ the Gramian's own origin): the law
  /// of cosines applied to the squared distance between any two of the
  /// simplex's vertices (vertex $0$ the Gramian's own origin). Generalizes
  /// [`Self::basis_angle`] (the case $v = 0$) to any of the simplex's
  /// $"dim"+1$ corners, not just the one the Gramian is based at.
  pub fn vertex_angle(&self, v: usize, a: usize, b: usize) -> f64 {
    let d_va = self.vertex_dist_sq(v, a);
    let d_vb = self.vertex_dist_sq(v, b);
    let d_ab = self.vertex_dist_sq(a, b);
    ((d_va + d_vb - d_ab) / (2.0 * (d_va * d_vb).sqrt())).acos()
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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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

/// Metric operations on tangent vectors (contravariant quantities), measured
/// by the metric tensor $g$. These are the canonical Riemannian measurements
/// of directions and lengths; the dual operations on covectors are available
/// through [`Self::covector_gramian`].
impl RiemannianMetric {
  /// Inner product $g(v, w)$ of two tangent vectors.
  pub fn inner(&self, v: &Vector, w: &Vector) -> f64 {
    self.vector_gramian.inner(v, w)
  }
  /// Squared length $g(v, v)$ of a tangent vector.
  pub fn norm_sq(&self, v: &Vector) -> f64 {
    self.vector_gramian.norm_sq(v)
  }
  /// Length $sqrt(g(v, v))$ of a tangent vector.
  pub fn norm(&self, v: &Vector) -> f64 {
    self.vector_gramian.norm(v)
  }
  /// Cosine of the angle between two tangent vectors.
  pub fn angle_cos(&self, v: &Vector, w: &Vector) -> f64 {
    self.vector_gramian.angle_cos(v, w)
  }
  /// Angle (in radians) between two tangent vectors.
  pub fn angle(&self, v: &Vector, w: &Vector) -> f64 {
    self.vector_gramian.angle(v, w)
  }
}

/// Inner product functionality on arbitrary elements.
impl Gramian {
  pub fn inner(&self, v: &Vector, w: &Vector) -> f64 {
    (v.transpose() * self.matrix() * w).x
  }
  /// Gram matrix of two families of vectors (given as columns): `vᵀ G w`.
  pub fn inner_mat(&self, v: &Matrix, w: &Matrix) -> Matrix {
    v.transpose() * self.matrix() * w
  }
  pub fn norm_sq(&self, v: &Vector) -> f64 {
    self.inner(v, v)
  }
  /// Elementwise squared norms of a family of vectors (given as columns).
  pub fn norm_sq_mat(&self, v: &Matrix) -> Matrix {
    self.inner_mat(v, v)
  }
  pub fn norm(&self, v: &Vector) -> f64 {
    self.inner(v, v).sqrt()
  }
  /// Elementwise norms of a family of vectors (given as columns).
  pub fn norm_mat(&self, v: &Matrix) -> Matrix {
    self.inner_mat(v, v).map(f64::sqrt)
  }
  /// Cosine of the angle between `v` and `w`.
  pub fn angle_cos(&self, v: &Vector, w: &Vector) -> f64 {
    self.inner(v, w) / self.norm(v) / self.norm(w)
  }
  /// Angle (in radians) between `v` and `w`.
  pub fn angle(&self, v: &Vector, w: &Vector) -> f64 {
    self.angle_cos(v, w).acos()
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::linalg::nalgebra::Vector;
  use std::f64::consts::FRAC_PI_2;

  #[test]
  fn euclidean_angles_and_norms() {
    let g = Gramian::standard(2);
    let e0 = Vector::from_column_slice(&[1.0, 0.0]);
    let e1 = Vector::from_column_slice(&[0.0, 1.0]);

    assert!((g.norm(&e0) - 1.0).abs() < 1e-12);
    assert!((g.angle(&e0, &e1) - FRAC_PI_2).abs() < 1e-12);
    assert!(g.angle_cos(&e0, &e1).abs() < 1e-12);
    assert!(g.angle(&e0, &e0).abs() < 1e-12);
  }

  #[test]
  fn nonstandard_metric_angle_matches_definition() {
    // A metric that stretches the second axis; the coordinate axes stay
    // g-orthogonal, but a diagonal vector no longer bisects them.
    let g = Gramian::new(Matrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 4.0]));
    let v = Vector::from_column_slice(&[1.0, 1.0]);
    let w = Vector::from_column_slice(&[1.0, 0.0]);

    // g(v, w) = 1, |v|_g = sqrt(5), |w|_g = 1.
    assert!((g.inner(&v, &w) - 1.0).abs() < 1e-12);
    assert!((g.norm(&v) - 5.0_f64.sqrt()).abs() < 1e-12);
    assert!((g.angle_cos(&v, &w) - 1.0 / 5.0_f64.sqrt()).abs() < 1e-12);
  }

  #[test]
  fn riemannian_metric_measures_tangent_vectors_with_g() {
    let g = Gramian::new(Matrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 3.0]));
    let metric = RiemannianMetric::new(g.clone());
    let v = Vector::from_column_slice(&[1.0, 1.0]);
    let w = Vector::from_column_slice(&[1.0, -1.0]);

    // Convenience methods on the metric agree with the tangent (vector) Gramian.
    assert!((metric.inner(&v, &w) - g.inner(&v, &w)).abs() < 1e-12);
    assert!((metric.norm(&v) - g.norm(&v)).abs() < 1e-12);
    assert!((metric.angle(&v, &w) - g.angle(&v, &w)).abs() < 1e-12);
  }
}
