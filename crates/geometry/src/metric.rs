pub mod manifold;

use index_algebra::IndexSet;
use itertools::Itertools;
use topology::Dim;

pub type EdgeIdx = usize;

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
