extern crate nalgebra as na;

pub mod field;
pub mod list;
pub mod term;

use common::{
  combo::binomial,
  linalg::nalgebra::{Matrix, Vector},
};
use term::{exterior_bases, ExteriorTerm};

pub type Dim = usize;
pub type ExteriorGrade = usize;

pub fn exterior_dim(dim: Dim, grade: ExteriorGrade) -> usize {
  binomial(dim, grade)
}

/// The exterior power functor $Lambda^k$ applied to a linear map.
///
/// Matrix of $Lambda^k A$ in the lexicographically ordered bases of $k$-blades:
/// the $k$-th compound matrix of all $k times k$ minors,
/// $(Lambda^k A)_(I J) = det A[I, J]$.
///
/// Functoriality $Lambda^k (A B) = (Lambda^k A)(Lambda^k B)$ is the
/// Cauchy-Binet formula. This single primitive induces the inner product on
/// $Lambda^k$ (multi_gramian), the pullback of $k$-forms (precompose_form)
/// and the coefficients of a wedge of vectors (spanning multivector).
pub fn exterior_power(linear_map: &Matrix, grade: ExteriorGrade) -> Matrix {
  let row_bases: Vec<_> = exterior_bases(linear_map.nrows(), grade).collect();
  let col_bases: Vec<_> = exterior_bases(linear_map.ncols(), grade).collect();

  let mut power = Matrix::zeros(row_bases.len(), col_bases.len());
  let mut minor = Matrix::zeros(grade, grade);
  for (i, row_basis) in row_bases.iter().enumerate() {
    for (j, col_basis) in col_bases.iter().enumerate() {
      for (ii, row) in row_basis.iter().enumerate() {
        for (jj, col) in col_basis.iter().enumerate() {
          minor[(ii, jj)] = linear_map[(row, col)];
        }
      }
      power[(i, j)] = minor.determinant();
    }
  }
  power
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
  /// Precompose k-form by some linear map: the pullback of differential
  /// k-forms.
  ///
  /// The contravariant functor $Lambda^k$ acting on coefficients,
  /// $c mapsto (Lambda^k A)^T c$.
  pub fn precompose_form(&self, linear_map: &Matrix) -> Self {
    assert_eq!(self.dim, linear_map.nrows());
    let coeffs = exterior_power(linear_map, self.grade).transpose() * &self.coeffs;
    Self::new(coeffs, linear_map.ncols(), self.grade)
  }

  pub fn apply_form_to_vector(&self, vector: &MultiVector) -> f64 {
    assert!(self.dim == vector.dim && self.grade == vector.grade);
    self.coeffs.dot(&vector.coeffs)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  use approx::assert_relative_eq;

  /// Deterministic full-rank-ish test matrix.
  fn test_matrix(nrows: usize, ncols: usize, seed: usize) -> Matrix {
    Matrix::from_fn(nrows, ncols, |i, j| {
      ((seed + 3 * i + 7 * j) % 5) as f64 / 5.0 + if i == j { 1.0 } else { 0.0 }
    })
  }

  /// Functoriality of the exterior power: the Cauchy-Binet formula
  /// $Lambda^k (A B) = (Lambda^k A)(Lambda^k B)$.
  #[test]
  fn exterior_power_functoriality() {
    for (n, m) in [(2, 2), (3, 3), (4, 4), (2, 3), (3, 2), (2, 4)] {
      let a = test_matrix(n, m, 1);
      let b = test_matrix(m, n, 2);
      for k in 0..=n.min(m) {
        let power_ab = exterior_power(&(&a * &b), k);
        let power_a_power_b = exterior_power(&a, k) * exterior_power(&b, k);
        assert_relative_eq!(power_ab, power_a_power_b, epsilon = 1e-12);
      }
    }
  }

  /// $Lambda^n A = det A$ and $Lambda^k id = id$.
  #[test]
  fn exterior_power_det_and_identity() {
    for n in 1..=4 {
      let a = test_matrix(n, n, 3);
      assert_relative_eq!(exterior_power(&a, n)[(0, 0)], a.determinant(), epsilon = 1e-12);
      for k in 0..=n {
        let id = Matrix::identity(n, n);
        assert_relative_eq!(
          exterior_power(&id, k),
          Matrix::identity(exterior_dim(n, k), exterior_dim(n, k))
        );
      }
    }
  }

  /// The pullback is the same as wedging the pulled-back constituent 1-forms.
  #[test]
  fn precompose_is_wedge_of_pullbacks() {
    for (n, m) in [(2, 2), (3, 3), (3, 2), (4, 3)] {
      let a = test_matrix(n, m, 4);
      for k in 0..=n.min(m) {
        let form = ExteriorElement::new(
          Vector::from_fn(exterior_dim(n, k), |i, _| (i + 1) as f64),
          n,
          k,
        );
        let computed = form.precompose_form(&a);
        let expected: ExteriorElement = form
          .basis_iter()
          .map(|(coeff, basis)| {
            coeff
              * MultiForm::wedge_big(basis.iter().map(|i| MultiForm::line(a.row(i).transpose())))
                .unwrap_or(ExteriorElement::one(m))
          })
          .sum();
        assert_relative_eq!(computed.coeffs(), expected.coeffs(), epsilon = 1e-12);
      }
    }
  }

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
