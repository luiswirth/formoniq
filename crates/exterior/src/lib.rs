extern crate nalgebra as na;

pub mod field;
pub mod term;
pub mod variance;

use term::{ExteriorBase, ExteriorTermExt, ScaledExteriorTerm};
use variance::VarianceMarker;

use multi_index::{binomial, variants::SetOrder, IndexSet};

use std::marker::PhantomData;

pub type Dim = usize;
pub type ExteriorGrade = usize;

#[derive(Debug, Clone)]
pub struct ExteriorElementList<V: VarianceMarker> {
  coeffs: na::DMatrix<f64>,
  dim: Dim,
  grade: ExteriorGrade,
  variance: PhantomData<V>,
}

impl<V: VarianceMarker> ExteriorElementList<V> {
  pub fn new(coeffs: na::DMatrix<f64>, dim: Dim, grade: ExteriorGrade) -> Self {
    assert_eq!(coeffs.nrows(), binomial(dim, grade));
    Self {
      coeffs,
      dim,
      grade,
      variance: PhantomData,
    }
  }

  pub fn dim(&self) -> Dim {
    self.dim
  }
  pub fn grade(&self) -> ExteriorGrade {
    self.grade
  }
  pub fn coeffs(&self) -> &na::DMatrix<f64> {
    &self.coeffs
  }
  pub fn into_coeffs(self) -> na::DMatrix<f64> {
    self.coeffs
  }
}

impl<V: VarianceMarker> FromIterator<ExteriorElement<V>> for ExteriorElementList<V> {
  fn from_iter<T: IntoIterator<Item = ExteriorElement<V>>>(iter: T) -> Self {
    let mut iter = iter.into_iter();
    let first = iter.next().unwrap();
    let dim = first.dim();
    let grade = first.grade();
    let mut coeffs = na::DMatrix::zeros(first.coeffs.len(), 1);
    coeffs.set_column(0, &first.coeffs);
    for (i, elem) in iter.enumerate() {
      assert!(elem.dim() == dim);
      assert!(elem.grade() == grade);
      coeffs = coeffs.insert_column(i + 1, 0.0);
      coeffs.set_column(i + 1, &elem.coeffs);
    }
    Self::new(coeffs, dim, grade)
  }
}

pub type MultiVectorList = ExteriorElementList<variance::Contra>;
pub type MultiFormList = ExteriorElementList<variance::Co>;

/// An element of an exterior algebra.
#[derive(Debug, Clone)]
pub struct ExteriorElement<V: VarianceMarker> {
  coeffs: na::DVector<f64>,
  dim: Dim,
  grade: ExteriorGrade,
  variance: PhantomData<V>,
}

impl<V: VarianceMarker> ExteriorElement<V> {
  pub fn new(coeffs: na::DVector<f64>, dim: Dim, grade: ExteriorGrade) -> Self {
    assert_eq!(coeffs.len(), binomial(dim, grade));
    Self {
      coeffs,
      dim,
      grade,
      variance: PhantomData,
    }
  }
  pub fn zero(dim: Dim, grade: ExteriorGrade) -> Self {
    Self {
      coeffs: na::DVector::zeros(binomial(dim, grade)),
      dim,
      grade,
      variance: PhantomData,
    }
  }
  pub fn one(dim: Dim) -> Self {
    Self {
      coeffs: na::DVector::from_element(1, 1.0),
      dim,
      grade: 0,
      variance: PhantomData,
    }
  }

  pub fn from_grade1(vector: na::DVector<f64>) -> Self {
    let dim = vector.len();
    Self::new(vector, dim, 1)
  }
  pub fn into_grade1(self) -> na::DVector<f64> {
    assert!(self.grade == 1);
    self.coeffs
  }

  pub fn dim(&self) -> Dim {
    self.dim
  }
  pub fn grade(&self) -> ExteriorGrade {
    self.grade
  }
  pub fn coeffs(&self) -> &na::DVector<f64> {
    &self.coeffs
  }
  pub fn into_coeffs(self) -> na::DVector<f64> {
    self.coeffs
  }

  pub fn basis_iter(&self) -> impl Iterator<Item = (f64, ExteriorBase<V>)> + use<'_, V> {
    let dim = self.dim;
    let grade = self.grade;
    self
      .coeffs
      .iter()
      .copied()
      .enumerate()
      .map(move |(i, coeff)| {
        let basis = IndexSet::from_lex_rank(dim, grade, i).ext(dim);
        (coeff, basis)
      })
  }

  pub fn basis_iter_mut(
    &mut self,
  ) -> impl Iterator<Item = (&mut f64, ExteriorBase<V>)> + use<'_, V> {
    let dim = self.dim;
    let grade = self.grade;
    self.coeffs.iter_mut().enumerate().map(move |(i, coeff)| {
      let basis = IndexSet::from_lex_rank(dim, grade, i).ext(dim);
      (coeff, basis)
    })
  }

  pub fn wedge(&self, other: &Self) -> Self {
    assert_eq!(self.dim, other.dim);
    let dim = self.dim;

    let new_grade = self.grade + other.grade;
    assert!(new_grade <= dim);

    let new_basis_size = binomial(self.dim, new_grade);
    let mut new_coeffs = na::DVector::zeros(new_basis_size);

    for (self_coeff, self_basis) in self.basis_iter() {
      for (other_coeff, other_basis) in other.basis_iter() {
        if self_basis == other_basis {
          continue;
        }
        if self_coeff == 0.0 || other_coeff == 0.0 {
          continue;
        }

        if let Some(merged_basis) = self_basis
          .indices()
          .clone()
          .union(other_basis.indices().clone())
          .try_into_sorted_signed()
        {
          let sign = merged_basis.sign;
          let merged_basis = merged_basis.set.lex_rank(dim);
          new_coeffs[merged_basis] += sign.as_f64() * self_coeff * other_coeff;
        }
      }
    }

    Self::new(new_coeffs, self.dim, new_grade)
  }

  pub fn wedge_big(factors: impl IntoIterator<Item = Self>) -> Option<Self> {
    let mut factors = factors.into_iter();
    let first = factors.next()?;
    let prod = factors.fold(first, |acc, factor| acc.wedge(&factor));
    Some(prod)
  }

  pub fn eq_epsilon(&self, other: &Self, eps: f64) -> bool {
    self.dim == other.dim
      && self.grade == other.grade
      && (&self.coeffs - &other.coeffs).norm_squared() <= eps.powi(2)
  }
}

impl<V: VarianceMarker> std::ops::Add<ExteriorElement<V>> for ExteriorElement<V> {
  type Output = Self;
  fn add(mut self, other: ExteriorElement<V>) -> Self::Output {
    self += other;
    self
  }
}
impl<V: VarianceMarker> std::ops::AddAssign<ExteriorElement<V>> for ExteriorElement<V> {
  fn add_assign(&mut self, other: ExteriorElement<V>) {
    assert_eq!(self.dim, other.dim);
    assert_eq!(self.grade, other.grade);
    self.coeffs += other.coeffs;
  }
}

impl<V: VarianceMarker> std::ops::Mul<f64> for ExteriorElement<V> {
  type Output = Self;
  fn mul(mut self, scalar: f64) -> Self::Output {
    self *= scalar;
    self
  }
}
impl<V: VarianceMarker> std::ops::MulAssign<f64> for ExteriorElement<V> {
  fn mul_assign(&mut self, scalar: f64) {
    self.coeffs *= scalar;
  }
}
impl<V: VarianceMarker> std::ops::Mul<ExteriorElement<V>> for f64 {
  type Output = ExteriorElement<V>;
  fn mul(self, rhs: ExteriorElement<V>) -> Self::Output {
    rhs * self
  }
}

impl<V: VarianceMarker, O: SetOrder> std::ops::AddAssign<ScaledExteriorTerm<V, O>>
  for ExteriorElement<V>
{
  fn add_assign(&mut self, term: ScaledExteriorTerm<V, O>) {
    let term = term.into_canonical();
    self[term.term] += term.coeff;
  }
}

impl<V: VarianceMarker, O: SetOrder> From<ScaledExteriorTerm<V, O>> for ExteriorElement<V> {
  fn from(term: ScaledExteriorTerm<V, O>) -> Self {
    let term = term.into_canonical();
    let mut element = Self::zero(term.dim(), term.grade());
    element[term.term] += term.coeff;
    element
  }
}

impl<V: VarianceMarker> std::ops::Index<ExteriorBase<V>> for ExteriorElement<V> {
  type Output = f64;
  fn index(&self, index: ExteriorBase<V>) -> &Self::Output {
    assert!(index.grade() == self.grade);
    let index = index.indices().lex_rank(self.dim);
    &self.coeffs[index]
  }
}
impl<V: VarianceMarker> std::ops::IndexMut<ExteriorBase<V>> for ExteriorElement<V> {
  fn index_mut(&mut self, index: ExteriorBase<V>) -> &mut Self::Output {
    let index = index.indices().lex_rank(self.dim);
    &mut self.coeffs[index]
  }
}
impl<V: VarianceMarker> std::ops::Index<usize> for ExteriorElement<V> {
  type Output = f64;
  fn index(&self, index: usize) -> &Self::Output {
    &self.coeffs[index]
  }
}

impl<V: VarianceMarker, O: SetOrder> std::iter::FromIterator<ScaledExteriorTerm<V, O>>
  for ExteriorElement<V>
{
  fn from_iter<T: IntoIterator<Item = ScaledExteriorTerm<V, O>>>(iter: T) -> Self {
    let mut iter = iter.into_iter();
    let first = iter.next().unwrap();
    let mut element = Self::from(first);
    iter.for_each(|term| element += term);
    element
  }
}
impl<V: VarianceMarker> std::iter::Sum for ExteriorElement<V> {
  fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
    let mut iter = iter.into_iter();
    let mut sum = iter.next().unwrap();
    for element in iter {
      sum += element;
    }
    sum
  }
}

pub type MultiVector = ExteriorElement<variance::Contra>;
impl MultiVector {}

pub type MultiForm = ExteriorElement<variance::Co>;
impl MultiForm {
  pub fn standard_volume_form(dim: Dim) -> Self {
    let grade = dim;
    let coeff = na::dvector![1.0];
    Self::new(coeff, dim, grade)
  }

  pub fn volume_form(gramian: &na::DMatrix<f64>) -> Self {
    let dim = gramian.nrows();
    let det_sqrt = gramian.determinant().sqrt();
    det_sqrt * Self::standard_volume_form(dim)
  }

  /// Precompose k-form by some linear map.
  ///
  /// Needed for pullback/pushforward of differential k-form.
  pub fn precompose(&self, linear_map: &na::DMatrix<f64>) -> Self {
    self
      .basis_iter()
      .map(|(coeff, basis)| {
        coeff
          * MultiForm::wedge_big(
            basis
              .indices()
              .iter()
              .map(|i| MultiForm::from_grade1(linear_map.row(i).transpose())),
          )
          .unwrap_or(ExteriorElement::one(self.dim))
      })
      .sum()
  }

  pub fn on_multivector(&self, kvector: &MultiVector) -> f64 {
    assert!(self.dim == kvector.dim && self.grade == kvector.grade);
    self.coeffs.dot(&kvector.coeffs)
  }

  pub fn hodge_star(&self, gramian: &na::DMatrix<f64>) -> Self {
    self
      .basis_iter()
      .map(|(coeff, basis)| (coeff * basis).hodge_star(gramian))
      .fold(Self::one(self.dim), |acc, a| acc + a)
  }
}

pub struct SimpleWedge<V: VarianceMarker> {
  factors: na::DMatrix<f64>,
  variance: PhantomData<V>,
}

impl<V: VarianceMarker> SimpleWedge<V> {
  pub fn new(factors: na::DMatrix<f64>) -> Self {
    Self {
      factors,
      variance: PhantomData,
    }
  }
  pub fn det(&self) -> f64 {
    self.factors.determinant()
  }
  pub fn vol(&self) -> f64 {
    self.det().abs()
  }
}

pub type VectorWedge = SimpleWedge<variance::Contra>;
impl VectorWedge {}

pub type CovectorWedge = SimpleWedge<variance::Co>;
impl CovectorWedge {
  pub fn evaluate(&self, vectors: &VectorWedge) -> f64 {
    let covectors = self;
    let mut mat = na::DMatrix::zeros(covectors.factors.len(), vectors.factors.len());
    for (i, covector) in covectors.factors.column_iter().enumerate() {
      for (j, vector) in vectors.factors.column_iter().enumerate() {
        mat[(i, j)] = covector.dot(&vector);
      }
    }
    mat.determinant()
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn compute_wedge() {
    let a = MultiForm::new(na::dvector![1.0, 0.0, 0.0], 3, 1);
    let b = MultiForm::new(na::dvector![0.0, 1.0, 0.0], 3, 1);

    let computed_ab = a.wedge(&b);
    let expected_ab = MultiForm::new(na::dvector![1.0, 0.0, 0.0], 3, 2);
    assert_eq!(computed_ab.coeffs, expected_ab.coeffs);
  }

  #[test]
  fn wedge_antisymmetry() {
    let form_a = MultiForm::new(na::dvector![1.0, 0.0, 0.0], 3, 1);
    let form_b = MultiForm::new(na::dvector![0.0, 1.0, 0.0], 3, 1);

    let ab = form_a.wedge(&form_b);
    let ba = form_b.wedge(&form_a);
    assert_eq!(ab.coeffs, -ba.coeffs);
  }

  #[test]
  fn wedge_with_zero() {
    let form = MultiForm::new(na::dvector![1.0, 0.0, 0.0], 3, 1);
    let zero_form = MultiForm::zero(3, 1);

    let wedge_computed = form.wedge(&zero_form);
    let wedge_expected = MultiForm::zero(3, 2);
    assert_eq!(wedge_computed.coeffs, wedge_expected.coeffs);
  }

  #[test]
  fn wedge_grade_exceeds_dim() {
    let a = MultiForm::new(na::dvector![1.0, 0.0], 2, 1);
    let b = MultiForm::new(na::dvector![1.0], 2, 2);
    let result = std::panic::catch_unwind(|| a.wedge(&b));
    assert!(result.is_err());
  }
}
