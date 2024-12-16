use crate::RiemannianMetric;

use common::Dim;
use index_algebra::IndexSet;

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
        let eij = IndexSet::from([vi, vj])
          .assume_sorted()
          .with_local_base(nvertices)
          .lex_rank();
        let lij = self[eij];

        let val = 0.5 * (l0i.powi(2) + l0j.powi(2) - lij.powi(2));

        metric_tensor[(i, j)] = val;
        metric_tensor[(j, i)] = val;
      }
    }
    RiemannianMetric::new(metric_tensor)
  }
}
