use crate::{
  variance::{self, VarianceMarker},
  Dim, ExteriorElement, ExteriorGrade, MultiForm, MultiFormList,
};

use common::metric::RiemannianMetric;
use multi_index::{
  combinators::{IndexSubPermutations, IndexSubsets},
  variants::{IncreasingSet, IndexKind},
  MultiIndex,
};

use std::marker::PhantomData;

pub type ExteriorBase<V> = ExteriorTerm<V, IncreasingSet>;

pub type MultiVectorTerm<O> = ExteriorTerm<variance::Contra, O>;
pub type MultiFormTerm<O> = ExteriorTerm<variance::Co, O>;

pub type MultiVectorBase = ExteriorBase<variance::Contra>;
pub type MultiFormBase = ExteriorBase<variance::Co>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExteriorTerm<V: VarianceMarker, O: IndexKind> {
  indices: MultiIndex<O>,
  dim: Dim,
  variance: PhantomData<V>,
}

impl<V: VarianceMarker, O: IndexKind> ExteriorTerm<V, O> {
  pub fn new(indices: MultiIndex<O>, dim: Dim) -> Self {
    Self {
      indices,
      dim,
      variance: PhantomData,
    }
  }
  pub fn indices(&self) -> &MultiIndex<O> {
    &self.indices
  }
  pub fn grade(&self) -> ExteriorGrade {
    self.indices.len()
  }
  pub fn dim(&self) -> Dim {
    self.dim
  }
}

impl MultiFormBase {
  // TODO: is there a more efficent implementation?
  /// Gramian must be inverse metric tensor for multiforms.
  /// For multivectors it's the regular metric tensor.
  pub fn hodge_star(&self, gramian: &na::DMatrix<f64>) -> MultiForm {
    let dim = self.dim();
    let grade = self.grade();
    let dual_grade = dim - grade;

    let primal_coeff = 1.0;
    let primal_index = &self.indices;

    let mut dual_element = ExteriorElement::zero(dim, dual_grade);
    for dual_index in IndexSubsets::canonical(dim, dual_grade) {
      let mut dual_coeff = 0.0;

      for sum_index in IndexSubPermutations::canonical(dim, grade) {
        let full_dual_index = sum_index.clone().union(dual_index.clone());
        let Some(full_dual_index) = full_dual_index.try_into_sorted_signed() else {
          // Levi-Civita symbol is zero.
          continue;
        };
        let sign = full_dual_index.sign.as_f64();

        let metric_prod: f64 = (0..grade)
          .map(|iindex| gramian[(primal_index[iindex], sum_index[iindex])])
          .product();

        dual_coeff += sign * primal_coeff * metric_prod;
      }
      dual_element += dual_coeff * dual_index.ext(dim);
    }

    gramian.determinant().sqrt() * dual_element
  }
}

#[derive(Debug, Clone)]
pub struct ScaledExteriorTerm<V: VarianceMarker, O: IndexKind> {
  pub coeff: f64,
  pub term: ExteriorTerm<V, O>,
}

pub type ScaledExteriorBase<V> = ScaledExteriorTerm<V, IncreasingSet>;

impl<V: VarianceMarker, O: IndexKind> ScaledExteriorTerm<V, O> {
  pub fn new(coeff: f64, term: ExteriorTerm<V, O>) -> Self {
    Self { coeff, term }
  }
  pub fn dim(&self) -> Dim {
    self.term.dim()
  }
  pub fn grade(&self) -> ExteriorGrade {
    self.term.grade()
  }

  pub fn into_canonical(self) -> ScaledExteriorTerm<V, IncreasingSet> {
    let dim = self.dim();
    let (term, sign) = self.term.indices.into_sorted_signed().into_parts();
    let coeff = self.coeff * sign.as_f64();
    let term = term.ext(dim);
    ScaledExteriorTerm { coeff, term }
  }

  pub fn assume_canonical(self) -> ScaledExteriorTerm<V, IncreasingSet> {
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

pub type ScaledMultiFormBase = ScaledExteriorBase<variance::Co>;
impl ScaledMultiFormBase {
  pub fn hodge_star(&self, gramian: &na::DMatrix<f64>) -> MultiForm {
    self.coeff * self.term.hodge_star(gramian)
  }
}

impl<V: VarianceMarker, O: IndexKind> std::ops::Mul<ExteriorTerm<V, O>> for f64 {
  type Output = ScaledExteriorTerm<V, O>;
  fn mul(self, term: ExteriorTerm<V, O>) -> Self::Output {
    let coeff = self;
    let term = term.indices.ext(term.dim);
    ScaledExteriorTerm::new(coeff, term)
  }
}

impl<V: VarianceMarker, O: IndexKind> std::ops::Mul<ScaledExteriorTerm<V, O>> for f64 {
  type Output = ScaledExteriorTerm<V, O>;
  fn mul(self, mut term: ScaledExteriorTerm<V, O>) -> Self::Output {
    term.coeff *= self;
    term
  }
}
impl<V: VarianceMarker, O: IndexKind> std::ops::MulAssign<f64> for ScaledExteriorTerm<V, O> {
  fn mul_assign(&mut self, scalar: f64) {
    self.coeff *= scalar;
  }
}

pub trait ExteriorTermExt<V: VarianceMarker, O: IndexKind> {
  fn ext(self, dim: Dim) -> ExteriorTerm<V, O>;
}
impl<V: VarianceMarker, Set: Into<MultiIndex<O>>, O: IndexKind> ExteriorTermExt<V, O> for Set {
  fn ext(self, dim: Dim) -> ExteriorTerm<V, O> {
    ExteriorTerm::new(self.into(), dim)
  }
}

pub trait RiemannianMetricExt {
  fn multi_form_gramian(&self, k: ExteriorGrade) -> na::DMatrix<f64>;
  fn multi_form_inner_product_mat(&self, v: &MultiFormList, w: &MultiFormList) -> na::DMatrix<f64>;
  fn multi_form_inner_product(&self, v: &MultiForm, w: &MultiForm) -> f64;
  fn multi_form_norm(&self, v: &MultiForm) -> f64;
}

// TODO: consider storing
/// Gram matrix on lexicographically ordered standard k-form standard basis.
impl RiemannianMetricExt for RiemannianMetric {
  fn multi_form_gramian(&self, k: ExteriorGrade) -> na::DMatrix<f64> {
    let n = self.dim();
    let combinations: Vec<_> = IndexSubsets::canonical(n, k).collect();
    let covector_gramian = self.covector_gramian();

    let mut multi_form_gramian = na::DMatrix::zeros(combinations.len(), combinations.len());
    let mut multi_basis_mat = na::DMatrix::zeros(k, k);

    for icomb in 0..combinations.len() {
      let combi = &combinations[icomb];
      for jcomb in icomb..combinations.len() {
        let combj = &combinations[jcomb];

        for iicomb in 0..k {
          let combii = combi[iicomb];
          for jjcomb in 0..k {
            let combjj = combj[jjcomb];
            multi_basis_mat[(iicomb, jjcomb)] = covector_gramian[(combii, combjj)];
          }
        }
        let det = multi_basis_mat.determinant();
        multi_form_gramian[(icomb, jcomb)] = det;
        multi_form_gramian[(jcomb, icomb)] = det;
      }
    }
    multi_form_gramian
  }

  fn multi_form_inner_product_mat(&self, v: &MultiFormList, w: &MultiFormList) -> na::DMatrix<f64> {
    assert_eq!(v.dim(), w.dim());
    assert_eq!(v.grade(), w.grade());
    v.coeffs().transpose() * self.multi_form_gramian(v.grade()) * w.coeffs()
  }

  fn multi_form_inner_product(&self, v: &MultiForm, w: &MultiForm) -> f64 {
    assert_eq!(v.dim(), w.dim());
    assert_eq!(v.grade(), w.grade());
    (v.coeffs().transpose() * self.multi_form_gramian(v.grade()) * w.coeffs()).x
  }

  fn multi_form_norm(&self, v: &MultiForm) -> f64 {
    self.multi_form_inner_product(v, v)
  }
}

#[cfg(test)]
mod test {
  use common::{linalg::assert_mat_eq, metric::RiemannianMetric};
  use multi_index::binomial;

  use super::RiemannianMetricExt;

  #[test]
  fn canonical_conversion() {
    use super::*;

    let dim = 4;
    let mut e0 = MultiForm::zero(dim, 3);
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
      let metric = na::DMatrix::identity(dim, dim);

      let primal = MultiIndex::new(vec![]).assume_sorted().ext(dim);
      let dual = primal.hodge_star(&metric);

      let expected_dual = 1.0 * MultiIndex::increasing(dim).ext(dim);
      let expected_dual = ExteriorElement::from(expected_dual);
      assert!(dual.eq_epsilon(&expected_dual, 10e-12));
    }
  }

  #[test]
  fn multi_form_gramian_euclidean() {
    for n in 0..=3 {
      let metric = RiemannianMetric::standard(n);
      for k in 0..=n {
        let binomial = binomial(n, k);
        let expected_gram = na::DMatrix::identity(binomial, binomial);
        let computed_gram = metric.multi_form_gramian(k);
        assert_mat_eq(&computed_gram, &expected_gram, None);
      }
    }
  }
}
