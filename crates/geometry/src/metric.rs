pub mod manifold;

use common::linalg::DMatrixExt;
use index_algebra::IndexSet;
use itertools::Itertools;
use topology::Dim;

pub type EdgeIdx = usize;

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

  pub fn from_tangent_vectors(tangent_vectors: na::DMatrix<f64>) -> Self {
    let metric_tensor = tangent_vectors.gramian();
    Self::new(metric_tensor)
  }

  /// Orthonormal metric.
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

/// Global or local list of edge lengths.
#[derive(Debug, Clone)]
pub struct EdgeLengths {
  vector: na::DVector<f64>,
}
impl EdgeLengths {
  pub fn new(vector: na::DVector<f64>) -> Self {
    Self { vector }
  }
  pub fn nedges(&self) -> usize {
    self.vector.len()
  }
  pub fn length(&self, iedge: EdgeIdx) -> f64 {
    self[iedge]
  }
  pub fn vector(&self) -> &na::DVector<f64> {
    &self.vector
  }
  pub fn vector_mut(&mut self) -> &mut na::DVector<f64> {
    &mut self.vector
  }
  pub fn into_vector(self) -> na::DVector<f64> {
    self.vector
  }
  pub fn iter(
    &self,
  ) -> na::iter::MatrixIter<
    '_,
    f64,
    na::Dyn,
    na::Const<1>,
    na::VecStorage<f64, na::Dyn, na::Const<1>>,
  > {
    self.vector.iter()
  }
}
impl std::ops::Index<EdgeIdx> for EdgeLengths {
  type Output = f64;
  fn index(&self, iedge: EdgeIdx) -> &Self::Output {
    &self.vector[iedge]
  }
}

impl EdgeLengths {
  pub fn restriction(&self, edges: impl Iterator<Item = EdgeIdx>) -> Self {
    let lengths = edges.map(|edge| self.length(edge)).collect_vec().into();
    Self::new(lengths)
  }

  /// Builds regge metric tensor from edge lenghts of simplex.
  ///
  /// On the simplicial manifold the edge vectors are the tangent vectors.
  pub fn to_regge_metric(&self, dim: Dim) -> RiemannianMetric {
    let nvertices = dim + 1;
    let mut metric_tensor = na::DMatrix::zeros(dim, dim);
    for i in 0..dim {
      metric_tensor[(i, i)] = self[i].powi(2);
    }
    for i in 0..dim {
      for j in (i + 1)..dim {
        let l0i = self[i];
        let l0j = self[j];

        let vi = i + 1;
        let vj = j + 1;
        // TODO: can we compute this more directly?
        let eij = IndexSet::from([vi, vj]).assume_sorted().lex_rank(nvertices);
        let lij = self[eij];

        let val = 0.5 * (l0i.powi(2) + l0j.powi(2) - lij.powi(2));

        metric_tensor[(i, j)] = val;
        metric_tensor[(j, i)] = val;
      }
    }
    RiemannianMetric::new(metric_tensor)
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
    let metric = RiemannianMetric::from_tangent_vectors(tangent_vectors.clone());
    let orthonormal_basis = metric.orthonormal_basis();
    assert!(
      (orthonormal_basis.transpose() * metric.metric_tensor * &orthonormal_basis
        - na::DMatrix::identity(dim, dim))
      .norm()
        <= 1e-12
    );

    let orthonormal_tangent_vectors = tangent_vectors * orthonormal_basis;
    let orthogonal_metric = RiemannianMetric::from_tangent_vectors(orthonormal_tangent_vectors);
    assert!((orthogonal_metric.metric_tensor() - na::DMatrix::identity(dim, dim)).norm() <= 1e-12);
  }
}
