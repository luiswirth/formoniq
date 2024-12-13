use super::{complex::Complex, simplicial::OrientedVertplex, RiemannianComplex};
use crate::{
  combo::combinators::IndexSubsets, exterior::ExteriorRank, mesh::complex::EdgeIdx,
  mesh::simplicial::OrderedVertplex, Dim,
};

pub struct RiemannianManifold {
  pub facets: Vec<OrientedVertplex>,
  pub edge_lengths: EdgeLengths,
}
impl RiemannianManifold {
  pub fn into_complex(self) -> RiemannianComplex {
    let Self {
      facets,
      edge_lengths,
    } = self;
    let complex = Complex::new(facets.clone());
    RiemannianComplex::new(facets, complex, edge_lengths)
  }
}

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
    self.vector[iedge]
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

  /// Builds regge metric tensor from edge lenghts of simplex cell.
  ///
  /// On the simplicial manifold the edge vectors are the tangent vectors.
  pub fn regge(dim: Dim, edge_lengths: &[f64]) -> Self {
    let nvertices = dim + 1;
    let mut metric_tensor = na::DMatrix::zeros(dim, dim);
    for i in 0..dim {
      metric_tensor[(i, i)] = edge_lengths[i].powi(2);
    }
    for i in 0..dim {
      for j in (i + 1)..dim {
        let l0i = edge_lengths[i];
        let l0j = edge_lengths[j];

        let vi = i + 1;
        let vj = j + 1;
        // TODO: can we compute this more directly?
        let eij = OrderedVertplex::from([vi, vj])
          .assume_sorted()
          .with_local_base(nvertices)
          .lex_rank();
        let lij = edge_lengths[eij];

        let val = 0.5 * (l0i.powi(2) + l0j.powi(2) - lij.powi(2));

        metric_tensor[(i, j)] = val;
        metric_tensor[(j, i)] = val;
      }
    }
    Self::new(metric_tensor)
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

  // TODO: consider storing
  /// Gram matrix on lexicographically ordered standard k-form standard basis.
  pub fn kform_gramian(&self, k: ExteriorRank) -> na::DMatrix<f64> {
    let n = self.dim();
    let combinations: Vec<_> = IndexSubsets::canonical(n, k).collect();
    let covector_gramian = self.covector_gramian();

    let mut kform_gramian = na::DMatrix::zeros(combinations.len(), combinations.len());
    let mut kbasis_mat = na::DMatrix::zeros(k, k);

    for icomb in 0..combinations.len() {
      let combi = &combinations[icomb];
      for jcomb in icomb..combinations.len() {
        let combj = &combinations[jcomb];

        for iicomb in 0..k {
          let combii = combi[iicomb];
          for jjcomb in 0..k {
            let combjj = combj[jjcomb];
            kbasis_mat[(iicomb, jjcomb)] = covector_gramian[(combii, combjj)];
          }
        }
        let det = kbasis_mat.determinant();
        kform_gramian[(icomb, jcomb)] = det;
        kform_gramian[(jcomb, icomb)] = det;
      }
    }
    kform_gramian
  }
  pub fn kform_inner_product(
    &self,
    k: ExteriorRank,
    v: &na::DMatrix<f64>,
    w: &na::DMatrix<f64>,
  ) -> na::DMatrix<f64> {
    v.transpose() * self.kform_gramian(k) * w
  }
  pub fn kform_norm_sqr(&self, k: ExteriorRank, v: &na::DMatrix<f64>) -> na::DMatrix<f64> {
    self.kform_inner_product(k, v, v)
  }
}

#[cfg(test)]
mod test {
  use super::RiemannianMetric;
  use crate::{combo::binomial, linalg::assert_mat_eq};

  #[test]
  fn kform_gramian_euclidean() {
    for n in 0..=3 {
      let metric = RiemannianMetric::euclidean(n);
      for k in 0..=n {
        let binomial = binomial(n, k);
        let expected_gram = na::DMatrix::identity(binomial, binomial);
        let computed_gram = metric.kform_gramian(k);
        assert_mat_eq(&computed_gram, &expected_gram);
      }
    }
  }
}
