//! Consider turning into self-contained (except for index-algebra) crate.

extern crate nalgebra as na;

use common::Dim;
use geometry::RiemannianMetric;
use index_algebra::{
  combinators::{IndexPermutations, IndexSubsets},
  sign::Sign,
  IndexSet,
};

use std::{collections::HashMap, mem};

pub type ExteriorRank = usize;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExteriorTerm {
  indices: IndexSet,
  dim: Dim,
}
impl ExteriorTerm {
  pub fn new(indices: impl Into<IndexSet>, dim: Dim) -> Self {
    let indices = indices.into();
    assert!(indices.len() <= dim);
    Self { indices, dim }
  }

  pub fn rank(&self) -> ExteriorRank {
    self.indices.len()
  }
  pub fn k(&self) -> ExteriorRank {
    self.rank()
  }
  pub fn dim(&self) -> Dim {
    self.dim
  }
  pub fn n(&self) -> Dim {
    self.dim()
  }

  pub fn canonicalize(&mut self) -> Sign {
    let mut indices = mem::take(&mut self.indices).signed(Sign::default());
    indices.sort();
    self.indices = indices.set;
    indices.sign
  }

  // TODO: is there a more efficent implementation?
  pub fn hodge_star(&self, metric: &RiemannianMetric) -> ExteriorElement {
    let n = self.n();
    let k = self.k();
    let dual_k = n - k;

    let primal_index = &self.indices;
    let primal_coeff = 1.0;

    let mut dual_terms = Vec::new();
    for dual_index in IndexSubsets::canonical(n, dual_k) {
      let mut dual_coeff = 0.0;

      for sum_index in IndexPermutations::canonical_sub(n, k) {
        let sum_index = sum_index.set;

        let full_dual_index = sum_index.clone().union(dual_index.clone());
        let mut full_dual_index = full_dual_index.signed(Sign::default());
        full_dual_index.sort();
        if full_dual_index.set.is_empty() {
          // Levi-Civita symbol is zero.
          continue;
        };

        let metric_prod: f64 = (0..k)
          .map(|iindex| metric.inverse_metric_tensor()[(primal_index[iindex], sum_index[iindex])])
          .product();

        let sign = full_dual_index.sign().as_f64();
        dual_coeff += sign * primal_coeff * metric_prod;
      }

      if dual_coeff != 0.0 {
        dual_terms.push(ScaledExteriorTerm::new(
          dual_coeff,
          Self::new(dual_index, self.dim),
        ));
      }
    }

    let dual_element = ExteriorElement::new(dual_terms);
    metric.det_sqrt() * dual_element
  }
}

#[derive(Debug, Clone)]
pub struct ScaledExteriorTerm {
  pub coeff: f64,
  pub term: ExteriorTerm,
}

impl ScaledExteriorTerm {
  pub fn new(coeff: f64, term: impl Into<ExteriorTerm>) -> Self {
    let term = term.into();
    Self { coeff, term }
  }

  pub fn canonicalize(&mut self) {
    let sign = self.term.canonicalize();
    self.coeff *= sign.as_f64();
  }

  pub fn eq_epsilon(&self, other: &Self, epsilon: f64) -> bool {
    self.term == other.term && (self.coeff - other.coeff).abs() < epsilon
  }
}

/// With Local Base and Sorted
impl ScaledExteriorTerm {
  pub fn hodge_star(&self, metric: &RiemannianMetric) -> ExteriorElement {
    self.coeff * self.term.hodge_star(metric)
  }
}

impl std::ops::Mul<ExteriorTerm> for f64 {
  type Output = ScaledExteriorTerm;
  fn mul(self, term: ExteriorTerm) -> Self::Output {
    ScaledExteriorTerm::new(self, term)
  }
}

impl std::ops::Mul<ScaledExteriorTerm> for f64 {
  type Output = ScaledExteriorTerm;
  fn mul(self, mut term: ScaledExteriorTerm) -> Self::Output {
    term.coeff *= self;
    term
  }
}
impl std::ops::MulAssign<f64> for ScaledExteriorTerm {
  fn mul_assign(&mut self, scalar: f64) {
    self.coeff *= scalar;
  }
}

impl std::ops::Mul<ExteriorElement> for f64 {
  type Output = ExteriorElement;
  fn mul(self, mut element: ExteriorElement) -> Self::Output {
    for term in &mut element.terms {
      *term *= self;
    }
    element
  }
}

#[derive(Debug, Clone)]
pub struct ExteriorElement {
  terms: Vec<ScaledExteriorTerm>,
  sorted: bool,
}

impl ExteriorElement {
  pub fn new(terms: Vec<ScaledExteriorTerm>) -> Self {
    let dim = terms[0].term.dim();
    let rank = terms[0].term.rank();
    assert!(terms.iter().all(|term| term.term.dim() == dim));
    assert!(terms.iter().all(|term| term.term.rank() == rank));
    Self {
      terms,
      sorted: false,
    }
  }
}

impl ExteriorElement {
  pub fn dim(&self) -> Dim {
    self.terms[0].term.dim()
  }
  pub fn n(&self) -> Dim {
    self.dim()
  }

  pub fn rank(&self) -> ExteriorRank {
    self.terms[0].term.rank()
  }
  pub fn k(&self) -> ExteriorRank {
    self.rank()
  }

  pub fn canonicalize(&mut self) {
    // TODO: remove empty terms
    for term in &mut self.terms {
      term.canonicalize();
    }
    self
      .terms
      .sort_unstable_by(|a, b| a.term.indices.pure_lexicographical_cmp(&b.term.indices));
    self.terms.dedup_by(|a, b| {
      if a.term == b.term {
        b.coeff += a.coeff;
        true
      } else {
        false
      }
    });

    // TODO: remove zero coeff terms

    self.sorted = true;
  }

  pub fn eq_epsilon(&self, other: &Self, epsilon: f64) -> bool {
    self
      .terms
      .iter()
      .zip(other.terms.iter())
      .all(|(a, b)| a.eq_epsilon(b, epsilon))
  }

  pub fn hodge_star(&self, metric: &RiemannianMetric) -> ExteriorElement {
    let mut dual_terms = HashMap::new();
    for primal_term in &self.terms {
      for dual_term in primal_term
        .clone()
        .with_base(self.dim())
        .hodge_star(metric)
        .terms
      {
        let dual_coeff = dual_terms.entry(dual_term.term).or_insert(0.0);
        *dual_coeff += dual_term.coeff;
      }
    }
    let dual_terms = dual_terms
      .into_iter()
      .map(|(term, coeff)| ScaledExteriorTerm::new(coeff, term))
      .collect();

    Self {
      terms: dual_terms,
      base: self.base,
      term_order: Ordered,
    }
  }
}

pub trait RiemannianMetricExt {
  fn kform_norm_sqr(&self, k: ExteriorRank, v: &na::DMatrix<f64>) -> na::DMatrix<f64>;
  fn kform_inner_product(
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
    v: &na::DMatrix<f64>,
    w: &na::DMatrix<f64>,
  ) -> na::DMatrix<f64> {
    v.transpose() * self.kform_gramian(k) * w
  }
  fn kform_norm_sqr(&self, k: ExteriorRank, v: &na::DMatrix<f64>) -> na::DMatrix<f64> {
    self.kform_inner_product(k, v, v)
  }
}

#[cfg(test)]
mod test {
  use common::linalg::assert_mat_eq;
  use geometry::RiemannianMetric;
  use index_algebra::binomial;

  use crate::RiemannianMetricExt;

  #[test]
  fn canonical_conversion() {
    use super::*;

    let terms = ExteriorElement::new(vec![
      ScaledExteriorTerm::new(1.0, vec![2, 0, 1]),
      ScaledExteriorTerm::new(3.0, vec![1, 3, 2]),
      ScaledExteriorTerm::new(-2.0, vec![0, 2, 1]),
      ScaledExteriorTerm::new(3.0, vec![0, 1, 2]),
    ]);
    let canonical = terms.canonicalize();
    let expected = ExteriorElement::new(vec![
      ScaledExteriorTerm::new(6.0, vec![0, 1, 2]),
      ScaledExteriorTerm::new(-3.0, vec![1, 2, 3]),
    ])
    .assume_canonical();
    assert!(canonical.eq_epsilon(&expected, 10e-12));
  }

  #[test]
  fn hodge_star_euclidean() {
    use super::*;

    for dim in 0..=3 {
      let metric = RiemannianMetric::euclidean(dim);

      let primal = ExteriorTerm::new(vec![])
        .assume_sorted()
        .with_local_base(dim)
        .ext();

      let dual = primal.hodge_star(&metric);

      let expected_dual = ExteriorTerm::increasing(dim).ext();
      let expected_dual = ExteriorElement::new(vec![1.0 * expected_dual]).assume_canonical();
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
