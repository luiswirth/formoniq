//! Consider turning into self-contained (except for index-algebra) crate.

extern crate nalgebra as na;

pub mod dense;

use common::Dim;
use geometry::RiemannianMetric;
use index_algebra::{
  combinators::{IndexPermutations, IndexSubsets},
  variants::*,
  IndexAlgebra,
};

use std::collections::HashMap;

pub type ExteriorRank = usize;

pub trait ExteriorTermExt<B: Base, O: Order, S: Signedness> {
  fn ext(self) -> ExteriorTerm<B, O, S>;
}
impl<B: Base, O: Order, S: Signedness> ExteriorTermExt<B, O, S> for IndexAlgebra<B, O, S> {
  fn ext(self) -> ExteriorTerm<B, O, S> {
    ExteriorTerm::new(self)
  }
}

impl<B: Base, O: Order, S: Signedness, T: Into<IndexAlgebra<B, O, S>>> From<T>
  for ExteriorTerm<B, O, S>
{
  fn from(value: T) -> Self {
    Self::new(value.into())
  }
}

pub type ExtBasis = ExteriorTerm<Unspecified, Sorted, Unsigned>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExteriorTerm<B: Base, O: Order, S: Signedness> {
  index_set: IndexAlgebra<B, O, S>,
}
impl<B: Base, O: Order, S: Signedness> ExteriorTerm<B, O, S> {
  pub fn new(index_set: IndexAlgebra<B, O, S>) -> Self {
    Self { index_set }
  }
  pub fn rank(&self) -> ExteriorRank {
    self.index_set.len()
  }
  pub fn k(&self) -> ExteriorRank {
    self.rank()
  }
}

impl<O: Order, S: Signedness> ExteriorTerm<Local, O, S> {
  pub fn dim(&self) -> Dim {
    self.index_set.base().len()
  }
  pub fn n(&self) -> Dim {
    self.dim()
  }
}

impl<S: Signedness> ExteriorTerm<Local, Sorted, S> {
  // TODO: is there a more efficent implementation?
  pub fn hodge_star(
    &self,
    metric: &RiemannianMetric,
  ) -> SparseExteriorElement<Local, Sorted, Sorted> {
    let n = self.n();
    let k = self.k();
    let dual_k = n - k;

    let primal_coeff = self.index_set.signedness().get_or_default().as_f64();
    let primal_index = self.index_set.clone().forget_sign();

    let mut dual_terms = Vec::new();
    for dual_index in IndexSubsets::canonical(n, dual_k) {
      let mut dual_coeff = 0.0;

      for sum_index in IndexPermutations::canonical_sub(n, k) {
        let sum_index = sum_index.forget_sign();
        let full_dual_index = sum_index.clone().union(dual_index.clone());
        let Some(full_dual_index) = full_dual_index.clone().try_sort_signed() else {
          // Levi-Civita symbol is zero.
          continue;
        };
        let sign = full_dual_index.sign().as_f64();

        let metric_prod: f64 = (0..k)
          .map(|iindex| metric.inverse_metric_tensor()[(primal_index[iindex], sum_index[iindex])])
          .product();

        dual_coeff += sign * primal_coeff * metric_prod;
      }

      if dual_coeff != 0.0 {
        dual_terms.push(ScaledExteriorTerm::new(
          dual_coeff,
          dual_index.clone().forget_base(),
        ));
      }
    }

    let dual_element = SparseExteriorElement {
      terms: dual_terms,
      base: *self.index_set.base(),
      term_order: Sorted,
    };

    metric.det_sqrt() * dual_element
  }
}

#[derive(Debug, Clone)]
pub struct ScaledExteriorTerm<B: Base, O: Order> {
  coeff: f64,
  term: ExteriorTerm<B, O, Unsigned>,
}

impl<B: Base, O: Order> ScaledExteriorTerm<B, O> {
  pub fn new(coeff: f64, term: impl Into<ExteriorTerm<B, O, Unsigned>>) -> Self {
    let term = term.into();
    Self { coeff, term }
  }

  pub fn coeff(&self) -> f64 {
    self.coeff
  }
  pub fn term(&self) -> &ExteriorTerm<B, O, Unsigned> {
    &self.term
  }

  pub fn rank(&self) -> ExteriorRank {
    self.term.rank()
  }
  pub fn k(&self) -> ExteriorRank {
    self.rank()
  }

  pub fn with_base(self, n: Dim) -> ScaledExteriorTerm<Local, O> {
    let coeff = self.coeff;
    let term = self.term.index_set.with_local_base(n).ext();
    ScaledExteriorTerm { coeff, term }
  }
  pub fn forget_base(self) -> ScaledExteriorTerm<Unspecified, O> {
    let coeff = self.coeff;
    let term = self.term.index_set.forget_base().ext();
    ScaledExteriorTerm { coeff, term }
  }

  pub fn into_canonical(self) -> ScaledExteriorTerm<B, Sorted> {
    let term = self.term.index_set.sort_signed();
    let coeff = self.coeff * term.sign().as_f64();
    let term = term.forget_sign().ext();
    ScaledExteriorTerm { coeff, term }
  }

  pub fn assume_canonical(self) -> ScaledExteriorTerm<B, Sorted> {
    ScaledExteriorTerm {
      coeff: self.coeff,
      term: self.term.index_set.assume_sorted().ext(),
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
impl<O: Order> ScaledExteriorTerm<Local, O> {
  pub fn dim(&self) -> Dim {
    self.term.index_set.base().len()
  }
  pub fn n(&self) -> Dim {
    self.dim()
  }
}

/// With Local Base and Sorted
impl ScaledExteriorTerm<Local, Sorted> {
  pub fn hodge_star(
    &self,
    metric: &RiemannianMetric,
  ) -> SparseExteriorElement<Local, Sorted, Sorted> {
    self.coeff * self.term.hodge_star(metric)
  }
}

impl<B: Base, O: Order, S: Signedness> std::ops::Mul<ExteriorTerm<B, O, S>> for f64 {
  type Output = ScaledExteriorTerm<B, O>;
  fn mul(self, term: ExteriorTerm<B, O, S>) -> Self::Output {
    let coeff = self * term.index_set.signedness().get_or_default().as_f64();
    let term = term.index_set.forget_sign();
    ScaledExteriorTerm::new(coeff, term)
  }
}

impl<B: Base, O: Order> std::ops::Mul<ScaledExteriorTerm<B, O>> for f64 {
  type Output = ScaledExteriorTerm<B, O>;
  fn mul(self, mut term: ScaledExteriorTerm<B, O>) -> Self::Output {
    term.coeff *= self;
    term
  }
}
impl<B: Base, O: Order> std::ops::MulAssign<f64> for ScaledExteriorTerm<B, O> {
  fn mul_assign(&mut self, scalar: f64) {
    self.coeff *= scalar;
  }
}

impl<B: Base, InnerO: Order, OuterO: Order> std::ops::Mul<SparseExteriorElement<B, InnerO, OuterO>>
  for f64
{
  type Output = SparseExteriorElement<B, InnerO, OuterO>;
  fn mul(self, mut element: SparseExteriorElement<B, InnerO, OuterO>) -> Self::Output {
    for term in &mut element.terms {
      *term *= self;
    }
    element
  }
}

#[derive(Debug, Clone)]
pub struct SparseExteriorElement<B: Base, InnerO: Order, OuterO: Order> {
  terms: Vec<ScaledExteriorTerm<Unspecified, InnerO>>,
  base: B,
  term_order: OuterO,
}

impl<B: Base, OInner: Order> SparseExteriorElement<B, OInner, Ordered> {
  pub fn new(terms: Vec<ScaledExteriorTerm<B, OInner>>) -> Self {
    let base = terms[0].term().index_set.base().clone();
    let rank = terms[0].rank();
    assert!(terms
      .iter()
      .all(|term| *term.term().index_set.base() == base));
    assert!(terms.iter().all(|term| term.rank() == rank));
    let terms = terms.into_iter().map(|t| t.forget_base()).collect();
    Self {
      terms,
      base,
      term_order: Ordered,
    }
  }
}

impl<B: Base, OInner: Order, OOuter: Order> SparseExteriorElement<B, OInner, OOuter> {
  pub fn rank(&self) -> ExteriorRank {
    self.terms[0].rank()
  }
  pub fn k(&self) -> ExteriorRank {
    self.rank()
  }

  pub fn term_order(&self) -> OOuter {
    self.term_order
  }

  pub fn into_canonical(self) -> SparseExteriorElement<B, Sorted, Sorted> {
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
      base: self.base,
      term_order: Sorted,
    }
  }

  pub fn assume_canonical(self) -> SparseExteriorElement<B, Sorted, Sorted> {
    let Self { terms, .. } = self;

    let terms: Vec<_> = terms
      .into_iter()
      .map(|term| term.assume_canonical())
      .collect();
    assert!(terms.is_sorted_by(|a, b| a.pure_lexicographical_cmp(b).is_lt()));

    SparseExteriorElement {
      terms,
      base: self.base,
      term_order: Sorted,
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

/// With Local Base
impl<OInner: Order, OOuter: Order> SparseExteriorElement<Local, OInner, OOuter> {
  pub fn dim(&self) -> Dim {
    self.base.len()
  }
  pub fn n(&self) -> Dim {
    self.dim()
  }
}

impl SparseExteriorElement<Local, Sorted, Ordered> {
  pub fn hodge_star(
    &self,
    metric: &RiemannianMetric,
  ) -> SparseExteriorElement<Local, Sorted, Ordered> {
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
  use geometry::RiemannianMetric;
  use index_algebra::binomial;

  use crate::RiemannianMetricExt;

  #[test]
  fn canonical_conversion() {
    use super::*;

    let terms = SparseExteriorElement::new(vec![
      ScaledExteriorTerm::new(1.0, vec![2, 0, 1]),
      ScaledExteriorTerm::new(3.0, vec![1, 3, 2]),
      ScaledExteriorTerm::new(-2.0, vec![0, 2, 1]),
      ScaledExteriorTerm::new(3.0, vec![0, 1, 2]),
    ]);
    let canonical = terms.into_canonical();
    let expected = SparseExteriorElement::new(vec![
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

      let primal = IndexAlgebra::new(vec![])
        .assume_sorted()
        .with_local_base(dim)
        .ext();

      let dual = primal.hodge_star(&metric);

      let expected_dual = IndexAlgebra::increasing(dim).ext();
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
