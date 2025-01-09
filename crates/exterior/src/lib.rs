//! Consider turning into self-contained (except for index-algebra) crate.

extern crate nalgebra as na;

pub mod dense;
pub mod manifold;

use geometry::metric::RiemannianMetric;
use index_algebra::{
  combinators::{IndexSubPermutations, IndexSubsets},
  variants::*,
  IndexSet,
};
use topology::Dim;

use std::collections::HashMap;

pub type ExteriorRank = usize;

pub type ExteriorBasis = ExteriorTerm<CanonicalOrder>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExteriorTerm<O: SetOrder> {
  index_set: IndexSet<O>,
  dim: Dim,
}

impl<O: SetOrder> ExteriorTerm<O> {
  pub fn new(index_set: IndexSet<O>, dim: Dim) -> Self {
    Self { index_set, dim }
  }
  pub fn rank(&self) -> ExteriorRank {
    self.index_set.len()
  }
  pub fn dim(&self) -> Dim {
    self.dim
  }
}

impl ExteriorBasis {
  // TODO: is there a more efficent implementation?
  pub fn hodge_star(
    &self,
    metric: &RiemannianMetric,
  ) -> SparseExteriorElement<CanonicalOrder, CanonicalOrder> {
    let n = self.dim();
    let k = self.rank();
    let dual_k = n - k;

    let primal_coeff = 1.0;
    let primal_index = &self.index_set;

    let mut dual_terms = Vec::new();
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

      if dual_coeff != 0.0 {
        dual_terms.push(ScaledExteriorTerm::new(dual_coeff, dual_index.ext(n)));
      }
    }

    let dual_element = SparseExteriorElement {
      terms: dual_terms,
      term_order: CanonicalOrder,
    };

    metric.det_sqrt() * dual_element
  }
}

#[derive(Debug, Clone)]
pub struct ScaledExteriorTerm<O: SetOrder> {
  coeff: f64,
  term: ExteriorTerm<O>,
}

impl<O: SetOrder> ScaledExteriorTerm<O> {
  pub fn new(coeff: f64, term: ExteriorTerm<O>) -> Self {
    Self { coeff, term }
  }

  pub fn coeff(&self) -> f64 {
    self.coeff
  }
  pub fn term(&self) -> &ExteriorTerm<O> {
    &self.term
  }

  pub fn rank(&self) -> ExteriorRank {
    self.term.rank()
  }
  pub fn k(&self) -> ExteriorRank {
    self.rank()
  }

  pub fn into_canonical(self) -> ScaledExteriorTerm<CanonicalOrder> {
    let dim = self.dim();
    let (term, sign) = self.term.index_set.into_sorted_signed().into_parts();
    let coeff = self.coeff * sign.as_f64();
    let term = term.ext(dim);
    ScaledExteriorTerm { coeff, term }
  }

  pub fn assume_canonical(self) -> ScaledExteriorTerm<CanonicalOrder> {
    ScaledExteriorTerm {
      coeff: self.coeff,
      term: self.term.index_set.assume_sorted().ext(self.term.dim),
    }
  }

  pub fn pure_lexicographical_cmp(&self, other: &Self) -> std::cmp::Ordering {
    self
      .term
      .index_set
      .pure_lexicographical_cmp(&other.term.index_set)
  }

  pub fn eq_epsilon(&self, other: &Self, epsilon: f64) -> bool {
    self.term == other.term && (self.coeff - other.coeff).abs() < epsilon
  }
}

/// With Local Base
impl<O: SetOrder> ScaledExteriorTerm<O> {
  pub fn dim(&self) -> Dim {
    self.term.dim()
  }
  pub fn n(&self) -> Dim {
    self.dim()
  }
}

/// With Local Base and Sorted
impl ScaledExteriorTerm<CanonicalOrder> {
  pub fn hodge_star(
    &self,
    metric: &RiemannianMetric,
  ) -> SparseExteriorElement<CanonicalOrder, CanonicalOrder> {
    self.coeff * self.term.hodge_star(metric)
  }
}

impl<O: SetOrder> std::ops::Mul<ExteriorTerm<O>> for f64 {
  type Output = ScaledExteriorTerm<O>;
  fn mul(self, term: ExteriorTerm<O>) -> Self::Output {
    let coeff = self;
    let term = term.index_set.ext(term.dim);
    ScaledExteriorTerm::new(coeff, term)
  }
}

impl<O: SetOrder> std::ops::Mul<ScaledExteriorTerm<O>> for f64 {
  type Output = ScaledExteriorTerm<O>;
  fn mul(self, mut term: ScaledExteriorTerm<O>) -> Self::Output {
    term.coeff *= self;
    term
  }
}
impl<O: SetOrder> std::ops::MulAssign<f64> for ScaledExteriorTerm<O> {
  fn mul_assign(&mut self, scalar: f64) {
    self.coeff *= scalar;
  }
}

impl<InnerO: SetOrder, OuterO: SetOrder> std::ops::Mul<SparseExteriorElement<InnerO, OuterO>>
  for f64
{
  type Output = SparseExteriorElement<InnerO, OuterO>;
  fn mul(self, mut element: SparseExteriorElement<InnerO, OuterO>) -> Self::Output {
    for term in &mut element.terms {
      *term *= self;
    }
    element
  }
}

#[derive(Debug, Clone)]
pub struct SparseExteriorElement<InnerO: SetOrder, OuterO: SetOrder> {
  terms: Vec<ScaledExteriorTerm<InnerO>>,
  term_order: OuterO,
}

impl<OInner: SetOrder> SparseExteriorElement<OInner, ArbitraryOrder> {
  pub fn new(terms: Vec<ScaledExteriorTerm<OInner>>) -> Self {
    let dim = terms[0].dim();
    let rank = terms[0].rank();
    assert!(terms.iter().all(|term| term.dim() == dim));
    assert!(terms.iter().all(|term| term.rank() == rank));
    Self {
      terms,
      term_order: ArbitraryOrder,
    }
  }
}

impl<OInner: SetOrder, OOuter: SetOrder> SparseExteriorElement<OInner, OOuter> {
  pub fn dim(&self) -> Dim {
    self.terms[0].dim()
  }
  pub fn rank(&self) -> ExteriorRank {
    self.terms[0].rank()
  }
  pub fn term_order(&self) -> OOuter {
    self.term_order
  }

  pub fn into_canonical(self) -> SparseExteriorElement<CanonicalOrder, CanonicalOrder> {
    let Self { terms, .. } = self;

    let mut terms: Vec<_> = terms
      .into_iter()
      .map(|term| term.into_canonical())
      .collect();
    terms.sort_unstable_by(ScaledExteriorTerm::pure_lexicographical_cmp);
    terms.dedup_by(|a, b| {
      if a.term == b.term {
        b.coeff += a.coeff;
        true
      } else {
        false
      }
    });

    SparseExteriorElement {
      terms,
      term_order: CanonicalOrder,
    }
  }

  pub fn assume_canonical(self) -> SparseExteriorElement<CanonicalOrder, CanonicalOrder> {
    let Self { terms, .. } = self;

    let terms: Vec<_> = terms
      .into_iter()
      .map(|term| term.assume_canonical())
      .collect();
    assert!(terms.is_sorted_by(|a, b| a.pure_lexicographical_cmp(b).is_lt()));

    SparseExteriorElement {
      terms,
      term_order: CanonicalOrder,
    }
  }

  pub fn eq_epsilon(&self, other: &Self, epsilon: f64) -> bool {
    self
      .terms
      .iter()
      .zip(other.terms.iter())
      .all(|(a, b)| a.eq_epsilon(b, epsilon))
  }
}

impl SparseExteriorElement<CanonicalOrder, ArbitraryOrder> {
  pub fn hodge_star(
    &self,
    metric: &RiemannianMetric,
  ) -> SparseExteriorElement<CanonicalOrder, ArbitraryOrder> {
    let mut dual_terms = HashMap::new();
    for primal_term in &self.terms {
      for dual_term in primal_term.clone().hodge_star(metric).terms {
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
      term_order: ArbitraryOrder,
    }
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

pub trait ExteriorTermExt<O: SetOrder> {
  fn ext(self, dim: Dim) -> ExteriorTerm<O>;
}
impl<Set: Into<IndexSet<O>>, O: SetOrder> ExteriorTermExt<O> for Set {
  fn ext(self, dim: Dim) -> ExteriorTerm<O> {
    ExteriorTerm::new(self.into(), dim)
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
    let terms = SparseExteriorElement::new(vec![
      ScaledExteriorTerm::new(1.0, vec![2, 0, 1].ext(dim)),
      ScaledExteriorTerm::new(3.0, vec![1, 3, 2].ext(dim)),
      ScaledExteriorTerm::new(-2.0, vec![0, 2, 1].ext(dim)),
      ScaledExteriorTerm::new(3.0, vec![0, 1, 2].ext(dim)),
    ]);
    let canonical = terms.into_canonical();
    let expected = SparseExteriorElement::new(vec![
      ScaledExteriorTerm::new(6.0, vec![0, 1, 2].ext(dim)),
      ScaledExteriorTerm::new(-3.0, vec![1, 2, 3].ext(dim)),
    ])
    .assume_canonical();
    assert!(canonical.eq_epsilon(&expected, 10e-12));
  }

  #[test]
  fn hodge_star_euclidean() {
    use super::*;

    for dim in 0..=3 {
      let metric = RiemannianMetric::euclidean(dim);

      let primal = IndexSet::new(vec![]).assume_sorted().ext(dim);

      let dual = primal.hodge_star(&metric);

      let expected_dual = IndexSet::increasing(dim).ext(dim);
      let expected_dual = SparseExteriorElement::new(vec![1.0 * expected_dual]).assume_canonical();
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
