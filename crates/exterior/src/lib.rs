extern crate nalgebra as na;

pub mod field;
pub mod list;
pub mod term;

use common::{
  combo::binomial,
  linalg::nalgebra::{Matrix, Vector},
};
use term::ExteriorTerm;

pub type Dim = usize;
pub type ExteriorGrade = usize;

pub fn exterior_dim(dim: Dim, grade: ExteriorGrade) -> usize {
  binomial(dim, grade)
}

/// An element of an exterior algebra.
#[derive(Debug, Clone)]
pub struct ExteriorElement {
  coeffs: Vector,
  dim: Dim,
  grade: ExteriorGrade,
}

impl ExteriorElement {
  pub fn new(coeffs: Vector, dim: Dim, grade: ExteriorGrade) -> Self {
    assert_eq!(coeffs.len(), exterior_dim(dim, grade));
    Self { coeffs, dim, grade }
  }

  pub fn scalar(v: f64, dim: Dim) -> ExteriorElement {
    Self::new(na::dvector![v], dim, 0)
  }
  pub fn line(coeffs: Vector) -> Self {
    let dim = coeffs.len();
    Self::new(coeffs, dim, 1)
  }

  pub fn zero(dim: Dim, grade: ExteriorGrade) -> Self {
    Self::new(Vector::zeros(exterior_dim(dim, grade)), dim, grade)
  }
  pub fn one(dim: Dim) -> Self {
    Self::scalar(1.0, dim)
  }

  pub fn into_grade1(self) -> Vector {
    assert!(self.grade == 1);
    self.coeffs
  }

  pub fn dim(&self) -> Dim {
    self.dim
  }
  pub fn grade(&self) -> ExteriorGrade {
    self.grade
  }
  pub fn coeffs(&self) -> &Vector {
    &self.coeffs
  }
  pub fn into_coeffs(self) -> Vector {
    self.coeffs
  }

  pub fn basis_iter(&self) -> impl Iterator<Item = (f64, ExteriorTerm)> + '_ {
    let dim = self.dim;
    let grade = self.grade;
    self
      .coeffs
      .iter()
      .copied()
      .enumerate()
      .map(move |(i, coeff)| {
        let basis = ExteriorTerm::from_lex_rank(dim, grade, i);
        (coeff, basis)
      })
  }

  pub fn basis_iter_mut(&mut self) -> impl Iterator<Item = (&mut f64, ExteriorTerm)> + '_ {
    let dim = self.dim;
    let grade = self.grade;
    self.coeffs.iter_mut().enumerate().map(move |(i, coeff)| {
      let basis = ExteriorTerm::from_lex_rank(dim, grade, i);
      (coeff, basis)
    })
  }

  pub fn wedge(&self, other: &Self) -> Self {
    assert_eq!(self.dim, other.dim);
    let dim = self.dim;

    let new_grade = self.grade + other.grade;
    if new_grade > dim {
      return Self::zero(dim, 0);
    }

    let new_basis_size = exterior_dim(dim, new_grade);
    let mut new_coeffs = Vector::zeros(new_basis_size);

    for (self_coeff, self_basis) in self.basis_iter() {
      for (other_coeff, other_basis) in other.basis_iter() {
        let self_basis = self_basis.clone();

        let coeff_prod = self_coeff * other_coeff;
        if self_basis == other_basis || coeff_prod == 0.0 {
          continue;
        }
        if let Some((sign, merged_basis)) = self_basis.wedge(other_basis).canonicalized() {
          let merged_basis = merged_basis.lex_rank();
          new_coeffs[merged_basis] += sign.as_f64() * coeff_prod;
        }
      }
    }

    Self::new(new_coeffs, dim, new_grade)
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

impl std::ops::Add<ExteriorElement> for ExteriorElement {
  type Output = Self;
  fn add(mut self, other: ExteriorElement) -> Self::Output {
    self += other;
    self
  }
}
impl std::ops::AddAssign<ExteriorElement> for ExteriorElement {
  fn add_assign(&mut self, other: ExteriorElement) {
    assert_eq!(self.dim, other.dim);
    assert_eq!(self.grade, other.grade);
    self.coeffs += other.coeffs;
  }
}

impl std::ops::Sub<ExteriorElement> for ExteriorElement {
  type Output = Self;
  fn sub(mut self, other: ExteriorElement) -> Self::Output {
    self -= other;
    self
  }
}
impl std::ops::SubAssign<ExteriorElement> for ExteriorElement {
  fn sub_assign(&mut self, other: ExteriorElement) {
    assert_eq!(self.dim, other.dim);
    assert_eq!(self.grade, other.grade);
    self.coeffs -= other.coeffs;
  }
}

impl std::ops::Mul<f64> for ExteriorElement {
  type Output = Self;
  fn mul(mut self, scalar: f64) -> Self::Output {
    self *= scalar;
    self
  }
}
impl std::ops::MulAssign<f64> for ExteriorElement {
  fn mul_assign(&mut self, scalar: f64) {
    self.coeffs *= scalar;
  }
}
impl std::ops::Mul<ExteriorElement> for f64 {
  type Output = ExteriorElement;
  fn mul(self, rhs: ExteriorElement) -> Self::Output {
    rhs * self
  }
}

impl std::ops::Index<ExteriorTerm> for ExteriorElement {
  type Output = f64;
  fn index(&self, term: ExteriorTerm) -> &Self::Output {
    assert!(
      term.is_basis(),
      "Can only index exterior element with exterior basis term."
    );
    assert!(term.dim() == self.dim());
    assert!(term.grade() == self.grade());
    &self.coeffs[term.lex_rank()]
  }
}
impl std::ops::IndexMut<ExteriorTerm> for ExteriorElement {
  fn index_mut(&mut self, term: ExteriorTerm) -> &mut Self::Output {
    assert!(
      term.is_basis(),
      "Can only index exterior element with exterior basis term."
    );
    assert!(term.dim() == self.dim());
    assert!(term.grade() == self.grade());
    &mut self.coeffs[term.lex_rank()]
  }
}
impl std::ops::Index<usize> for ExteriorElement {
  type Output = f64;
  fn index(&self, index: usize) -> &Self::Output {
    &self.coeffs[index]
  }
}

impl From<ExteriorTerm> for ExteriorElement {
  fn from(term: ExteriorTerm) -> Self {
    let mut element = Self::zero(term.dim(), term.grade());
    if let Some((sign, basis)) = term.canonicalized() {
      element[basis] = sign.as_f64();
    }
    element
  }
}

impl std::ops::AddAssign<ExteriorTerm> for ExteriorElement {
  fn add_assign(&mut self, term: ExteriorTerm) {
    self[term] += 1.0;
  }
}
impl std::ops::Add<ExteriorTerm> for ExteriorElement {
  type Output = ExteriorElement;
  fn add(mut self, term: ExteriorTerm) -> Self::Output {
    self += term;
    self
  }
}

impl std::iter::FromIterator<ExteriorTerm> for ExteriorElement {
  fn from_iter<T: IntoIterator<Item = ExteriorTerm>>(iter: T) -> Self {
    let mut iter = iter.into_iter();
    let first = iter.next().unwrap();
    let mut element = Self::from(first);
    iter.for_each(|term| element += term);
    element
  }
}
impl std::iter::Sum for ExteriorElement {
  fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
    let mut iter = iter.into_iter();
    let mut sum = iter.next().unwrap();
    for element in iter {
      sum += element;
    }
    sum
  }
}

pub type MultiVector = ExteriorElement;
pub type MultiForm = ExteriorElement;
impl MultiForm {
  /// Precompose k-form by some linear map.
  ///
  /// Needed for pullback of differential k-form.
  pub fn precompose_form(&self, linear_map: &Matrix) -> Self {
    self
      .basis_iter()
      .map(|(coeff, basis)| {
        coeff
          * MultiForm::wedge_big(
            basis
              .iter()
              .map(|i| MultiForm::line(linear_map.row(i).transpose())),
          )
          .unwrap_or(ExteriorElement::one(self.dim))
      })
      .sum()
  }

  pub fn apply_form_to_vector(&self, vector: &MultiVector) -> f64 {
    assert!(self.dim == vector.dim && self.grade == vector.grade);
    self.coeffs.dot(&vector.coeffs)
  }
}

pub struct SimpleWedge {
  factors: Matrix,
}

impl SimpleWedge {
  pub fn new(factors: Matrix) -> Self {
    Self { factors }
  }
  pub fn det(&self) -> f64 {
    self.factors.determinant()
  }
  pub fn vol(&self) -> f64 {
    self.det().abs()
  }
}

pub type VectorWedge = SimpleWedge;
impl VectorWedge {}

pub type CovectorWedge = SimpleWedge;
impl CovectorWedge {
  pub fn evaluate(&self, vectors: &VectorWedge) -> f64 {
    let covectors = self;
    let mut mat = Matrix::zeros(covectors.factors.len(), vectors.factors.len());
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
    let a = ExteriorElement::new(na::dvector![1.0, 0.0, 0.0], 3, 1);
    let b = ExteriorElement::new(na::dvector![0.0, 1.0, 0.0], 3, 1);

    let computed_ab = a.wedge(&b);
    let expected_ab = ExteriorElement::new(na::dvector![1.0, 0.0, 0.0], 3, 2);
    assert_eq!(computed_ab.coeffs, expected_ab.coeffs);
  }

  #[test]
  fn wedge_antisymmetry() {
    let form_a = ExteriorElement::new(na::dvector![1.0, 0.0, 0.0], 3, 1);
    let form_b = ExteriorElement::new(na::dvector![0.0, 1.0, 0.0], 3, 1);

    let ab = form_a.wedge(&form_b);
    let ba = form_b.wedge(&form_a);
    assert_eq!(ab.coeffs, -ba.coeffs);
  }

  #[test]
  fn wedge_with_zero() {
    let form = ExteriorElement::new(na::dvector![1.0, 0.0, 0.0], 3, 1);
    let zero_form = ExteriorElement::zero(3, 1);

    let wedge_computed = form.wedge(&zero_form);
    let wedge_expected = ExteriorElement::zero(3, 2);
    assert_eq!(wedge_computed.coeffs, wedge_expected.coeffs);
  }

  #[test]
  fn wedge_grade_exceeds_dim() {
    let a = ExteriorElement::new(na::dvector![1.0, 0.0], 2, 1);
    let b = ExteriorElement::new(na::dvector![1.0], 2, 2);
    let computed = a.wedge(&b);
    let expected = ExteriorElement::zero(2, 0);
    assert_eq!(computed.coeffs, expected.coeffs);
  }
}
