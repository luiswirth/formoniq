//! Consider turning into self-contained (except for index-algebra) crate.

extern crate nalgebra as na;

pub mod dense;
pub mod manifold;
pub mod variance;

use std::marker::PhantomData;

use dense::{ExteriorElement, KForm};
use geometry::metric::RiemannianMetric;
use index_algebra::{
  combinators::{IndexSubPermutations, IndexSubsets},
  variants::*,
  IndexSet,
};
use topology::Dim;
use variance::VarianceMarker;

pub type ExteriorRank = usize;
pub type ExteriorBasis<V> = ExteriorTerm<V, CanonicalOrder>;

pub type KVectorTerm<O> = ExteriorTerm<variance::Contra, O>;
pub type KFormTerm<O> = ExteriorTerm<variance::Co, O>;

pub type KVectorBasis = ExteriorBasis<variance::Contra>;
pub type KFormBasis = ExteriorBasis<variance::Co>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExteriorTerm<V: VarianceMarker, O: SetOrder> {
  indices: IndexSet<O>,
  dim: Dim,
  variance: PhantomData<V>,
}

impl<V: VarianceMarker, O: SetOrder> ExteriorTerm<V, O> {
  pub fn new(indices: IndexSet<O>, dim: Dim) -> Self {
    Self {
      indices,
      dim,
      variance: PhantomData,
    }
  }
  pub fn indices(&self) -> &IndexSet<O> {
    &self.indices
  }
  pub fn rank(&self) -> ExteriorRank {
    self.indices.len()
  }
  pub fn dim(&self) -> Dim {
    self.dim
  }
}

impl KFormBasis {
  // TODO: is there a more efficent implementation?
  pub fn hodge_star(&self, metric: &RiemannianMetric) -> KForm {
    let n = self.dim();
    let k = self.rank();
    let dual_k = n - k;

    let primal_coeff = 1.0;
    let primal_index = &self.indices;

    let mut dual_element = ExteriorElement::zero(n, dual_k);
    for dual_index in IndexSubsets::canonical(n, dual_k) {
      let mut dual_coeff = 0.0;

      for sum_index in IndexSubPermutations::canonical(n, k) {
        let full_dual_index = sum_index.clone().union(dual_index.clone());
        let Some(full_dual_index) = full_dual_index.try_into_sorted_signed() else {
          // Levi-Civita symbol is zero.
          continue;
        };
        let sign = full_dual_index.sign.as_f64();

        let metric_prod: f64 = (0..k)
          .map(|iindex| metric.inverse_metric_tensor()[(primal_index[iindex], sum_index[iindex])])
          .product();

        dual_coeff += sign * primal_coeff * metric_prod;
      }
      dual_element += dual_coeff * dual_index.ext(n);
    }

    metric.det_sqrt() * dual_element
  }
}

#[derive(Debug, Clone)]
pub struct ScaledExteriorTerm<V: VarianceMarker, O: SetOrder> {
  coeff: f64,
  term: ExteriorTerm<V, O>,
}

pub type ScaledExteriorBasis<V> = ScaledExteriorTerm<V, CanonicalOrder>;

impl<V: VarianceMarker, O: SetOrder> ScaledExteriorTerm<V, O> {
  pub fn new(coeff: f64, term: ExteriorTerm<V, O>) -> Self {
    Self { coeff, term }
  }

  pub fn coeff(&self) -> f64 {
    self.coeff
  }
  pub fn term(&self) -> &ExteriorTerm<V, O> {
    &self.term
  }
  pub fn dim(&self) -> Dim {
    self.term.dim()
  }
  pub fn rank(&self) -> ExteriorRank {
    self.term.rank()
  }

  pub fn into_canonical(self) -> ScaledExteriorTerm<V, CanonicalOrder> {
    let dim = self.dim();
    let (term, sign) = self.term.indices.into_sorted_signed().into_parts();
    let coeff = self.coeff * sign.as_f64();
    let term = term.ext(dim);
    ScaledExteriorTerm { coeff, term }
  }

  pub fn assume_canonical(self) -> ScaledExteriorTerm<V, CanonicalOrder> {
    ScaledExteriorTerm {
      coeff: self.coeff,
      term: self.term.indices.assume_sorted().ext(self.term.dim),
    }
  }

  pub fn pure_lexicographical_cmp(&self, other: &Self) -> std::cmp::Ordering {
    self
      .term
      .indices
      .pure_lexicographical_cmp(&other.term.indices)
  }

  pub fn eq_epsilon(&self, other: &Self, epsilon: f64) -> bool {
    self.term == other.term && (self.coeff - other.coeff).abs() < epsilon
  }
}

pub type ScaledKFormBasis = ScaledExteriorBasis<variance::Co>;
impl ScaledKFormBasis {
  pub fn hodge_star(&self, metric: &RiemannianMetric) -> KForm {
    self.coeff * self.term.hodge_star(metric)
  }
}

impl<V: VarianceMarker, O: SetOrder> std::ops::Mul<ExteriorTerm<V, O>> for f64 {
  type Output = ScaledExteriorTerm<V, O>;
  fn mul(self, term: ExteriorTerm<V, O>) -> Self::Output {
    let coeff = self;
    let term = term.indices.ext(term.dim);
    ScaledExteriorTerm::new(coeff, term)
  }
}

impl<V: VarianceMarker, O: SetOrder> std::ops::Mul<ScaledExteriorTerm<V, O>> for f64 {
  type Output = ScaledExteriorTerm<V, O>;
  fn mul(self, mut term: ScaledExteriorTerm<V, O>) -> Self::Output {
    term.coeff *= self;
    term
  }
}
impl<V: VarianceMarker, O: SetOrder> std::ops::MulAssign<f64> for ScaledExteriorTerm<V, O> {
  fn mul_assign(&mut self, scalar: f64) {
    self.coeff *= scalar;
  }
}

pub trait ExteriorTermExt<V: VarianceMarker, O: SetOrder> {
  fn ext(self, dim: Dim) -> ExteriorTerm<V, O>;
}
impl<V: VarianceMarker, Set: Into<IndexSet<O>>, O: SetOrder> ExteriorTermExt<V, O> for Set {
  fn ext(self, dim: Dim) -> ExteriorTerm<V, O> {
    ExteriorTerm::new(self.into(), dim)
  }
}

pub trait RiemannianMetricExt {
  fn kform_inner_product(&self, k: ExteriorRank, v: &na::DVector<f64>, w: &na::DVector<f64>)
    -> f64;
  fn kform_norm_sqr(&self, k: ExteriorRank, v: &na::DMatrix<f64>) -> na::DMatrix<f64>;
  fn kform_inner_product_mat(
    &self,
    k: ExteriorRank,
    v: &na::DMatrix<f64>,
    w: &na::DMatrix<f64>,
  ) -> na::DMatrix<f64>;
  fn kform_gramian(&self, k: ExteriorRank) -> na::DMatrix<f64>;
}

// TODO: consider storing
/// Gram matrix on lexicographically ordered standard k-form standard basis.
impl RiemannianMetricExt for RiemannianMetric {
  fn kform_gramian(&self, k: ExteriorRank) -> na::DMatrix<f64> {
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
  fn kform_inner_product(
    &self,
    k: ExteriorRank,
    v: &na::DVector<f64>,
    w: &na::DVector<f64>,
  ) -> f64 {
    (v.transpose() * self.kform_gramian(k) * w).x
  }

  fn kform_inner_product_mat(
    &self,
    k: ExteriorRank,
    v: &na::DMatrix<f64>,
    w: &na::DMatrix<f64>,
  ) -> na::DMatrix<f64> {
    println!("{:?}", v.shape());
    println!("{:?}", self.kform_gramian(k).shape());
    v.transpose() * self.kform_gramian(k) * w
  }
  fn kform_norm_sqr(&self, k: ExteriorRank, v: &na::DMatrix<f64>) -> na::DMatrix<f64> {
    self.kform_inner_product_mat(k, v, v)
  }
}

#[cfg(test)]
mod test {
  use common::linalg::assert_mat_eq;
  use geometry::metric::RiemannianMetric;
  use index_algebra::binomial;

  use crate::RiemannianMetricExt;

  #[test]
  fn canonical_conversion() {
    use super::*;

    let dim = 4;
    let mut e0 = KForm::zero(dim, 3);
    e0 += 1.0 * vec![2, 0, 1].ext(dim);
    e0 += 3.0 * vec![1, 3, 2].ext(dim);
    e0 += -2.0 * vec![0, 2, 1].ext(dim);
    e0 += 3.0 * vec![0, 1, 2].ext(dim);

    let mut e1 = ExteriorElement::zero(dim, 3);
    e1 += 6.0 * vec![0, 1, 2].ext(dim);
    e1 += -3.0 * vec![1, 2, 3].ext(dim);

    assert!(e0.eq_epsilon(&e1, 10e-12));
  }

  #[test]
  fn hodge_star_euclidean() {
    use super::*;

    for dim in 0..=3 {
      let metric = RiemannianMetric::euclidean(dim);

      let primal = IndexSet::new(vec![]).assume_sorted().ext(dim);
      let dual = primal.hodge_star(&metric);

      let expected_dual = 1.0 * IndexSet::increasing(dim).ext(dim);
      let expected_dual = ExteriorElement::from(expected_dual);
      assert!(dual.eq_epsilon(&expected_dual, 10e-12));
    }
  }

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
