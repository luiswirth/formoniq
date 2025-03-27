use crate::{
  variance::{self, VarianceMarker},
  Dim, ExteriorElement, ExteriorGrade, MultiForm, MultiFormList,
};

use common::metric::RiemannianMetric;
use multi_index::{
  binomial,
  sign::{sort_signed, Sign},
};

use std::marker::PhantomData;

pub type MultiVectorTerm = ExteriorTerm<variance::Contra>;
pub type MultiFormTerm = ExteriorTerm<variance::Co>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExteriorTerm<V: VarianceMarker> {
  indices: Vec<usize>,
  dim: Dim,
  variance: PhantomData<V>,
}

impl<V: VarianceMarker> ExteriorTerm<V> {
  pub fn new(indices: Vec<usize>, dim: Dim) -> Self {
    Self {
      indices,
      dim,
      variance: PhantomData,
    }
  }
  pub fn top(dim: Dim) -> Self {
    Self::new((0..dim).collect(), dim)
  }
  pub fn indices(&self) -> &[usize] {
    &self.indices
  }
  pub fn grade(&self) -> ExteriorGrade {
    self.indices.len()
  }
  pub fn dim(&self) -> Dim {
    self.dim
  }

  pub fn is_canonical(&self) -> bool {
    let Some((sign, canonical)) = self.clone().canonical() else {
      return false;
    };
    sign == Sign::Pos && canonical == *self
  }
  pub fn is_basis(&self) -> bool {
    self.is_canonical()
  }
  pub fn canonical(mut self) -> Option<(Sign, Self)> {
    let sign = sort_signed(&mut self.indices);
    let len = self.indices.len();
    self.indices.dedup();
    if self.indices.len() != len {
      return None;
    }
    Some((sign, self))
  }

  pub fn wedge(mut self, mut other: Self) -> Self {
    self.indices.append(&mut other.indices);
    self
  }

  pub fn iter(&self) -> std::iter::Copied<std::slice::Iter<'_, usize>> {
    self.indices.iter().copied()
  }
}

impl<V: VarianceMarker> std::ops::Index<usize> for ExteriorTerm<V> {
  type Output = usize;
  fn index(&self, index: usize) -> &Self::Output {
    &self.indices[index]
  }
}

impl<V: VarianceMarker> ExteriorTerm<V> {
  pub fn from_lex_rank(dim: Dim, grade: ExteriorGrade, mut rank: usize) -> Self {
    let mut indices = Vec::with_capacity(grade);
    let mut start = 0;
    for i in 0..grade {
      let remaining = grade - i;
      for x in start..=(dim - remaining) {
        let c = binomial(dim - x - 1, remaining - 1);
        if rank < c {
          indices.push(x);
          start = x + 1;
          break;
        } else {
          rank -= c;
        }
      }
    }
    Self::new(indices, dim)
  }

  pub fn from_graded_lex_rank(dim: Dim, grade: ExteriorGrade, mut rank: usize) -> Self {
    rank -= Self::graded_lex_rank_offset(dim, grade);
    Self::from_lex_rank(dim, grade, rank)
  }

  pub fn lex_rank(&self) -> usize {
    assert!(self.is_basis(), "Lex rank only available for basis terms.");
    let dim = self.dim();
    let grade = self.grade();

    let mut rank = 0;
    for (i, index) in self.iter().enumerate() {
      let start = if i == 0 { 0 } else { self[i - 1] + 1 };
      for s in start..index {
        rank += binomial(dim - s - 1, grade - i - 1);
      }
    }
    rank
  }

  pub fn graded_lex_rank(&self) -> usize {
    let dim = self.dim();
    let grade = self.grade();
    Self::graded_lex_rank_offset(dim, grade) + self.lex_rank()
  }

  fn graded_lex_rank_offset(dim: usize, grade: usize) -> usize {
    (0..grade).map(|s| binomial(dim, s)).sum()
  }
}

impl<V: VarianceMarker> std::ops::Mul<ExteriorTerm<V>> for f64 {
  type Output = ExteriorElement<V>;
  fn mul(self, term: ExteriorTerm<V>) -> Self::Output {
    let coeff = self;
    coeff * ExteriorElement::from(term)
  }
}

pub fn exterior_bases<V: VarianceMarker>(
  dim: Dim,
  grade: ExteriorGrade,
) -> impl Iterator<Item = ExteriorTerm<V>> {
  itertools::Itertools::combinations(0..dim, grade)
    .map(move |indices| ExteriorTerm::new(indices, dim))
}

pub fn exterior_terms<V: VarianceMarker>(
  dim: Dim,
  grade: ExteriorGrade,
) -> impl Iterator<Item = ExteriorTerm<V>> {
  itertools::Itertools::permutations(0..dim, grade)
    .map(move |indices| ExteriorTerm::new(indices, dim))
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
    let bases: Vec<_> = exterior_bases::<variance::Co>(n, k).collect();
    let covector_gramian = self.covector_gramian();

    let mut multi_form_gramian = na::DMatrix::zeros(bases.len(), bases.len());
    let mut multi_basis_mat = na::DMatrix::zeros(k, k);

    for icomb in 0..bases.len() {
      let combi = &bases[icomb];
      for jcomb in icomb..bases.len() {
        let combj = &bases[jcomb];

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
    use super::ExteriorTerm as Ext;
    use super::*;

    let dim = 4;
    let mut e0 = MultiForm::zero(dim, 3);
    e0 += 1.0 * Ext::new(vec![2, 0, 1], dim);
    e0 += 3.0 * Ext::new(vec![1, 3, 2], dim);
    e0 += -2.0 * Ext::new(vec![0, 2, 1], dim);
    e0 += 3.0 * Ext::new(vec![0, 1, 2], dim);

    let mut e1 = ExteriorElement::zero(dim, 3);
    e1 += 6.0 * Ext::new(vec![0, 1, 2], dim);
    e1 += -3.0 * Ext::new(vec![1, 2, 3], dim);

    assert!(e0.eq_epsilon(&e1, 10e-12));
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
