use crate::{Dim, ExteriorElement, ExteriorGrade};

use common::{combo::binomial, linalg::nalgebra::Matrix};

pub type MultiVectorList = ExteriorElementList;
pub type MultiFormList = ExteriorElementList;

#[derive(Debug, Clone)]
pub struct ExteriorElementList {
  coeffs: Matrix,
  dim: Dim,
  grade: ExteriorGrade,
}

impl ExteriorElementList {
  pub fn new(coeffs: Matrix, dim: Dim, grade: ExteriorGrade) -> Self {
    assert_eq!(coeffs.nrows(), binomial(dim, grade));
    Self { coeffs, dim, grade }
  }

  pub fn dim(&self) -> Dim {
    self.dim
  }
  pub fn grade(&self) -> ExteriorGrade {
    self.grade
  }
  pub fn coeffs(&self) -> &Matrix {
    &self.coeffs
  }
  pub fn into_coeffs(self) -> Matrix {
    self.coeffs
  }
}

impl FromIterator<ExteriorElement> for ExteriorElementList {
  fn from_iter<T: IntoIterator<Item = ExteriorElement>>(iter: T) -> Self {
    let elements: Vec<_> = iter.into_iter().collect();
    let first = elements.first().expect("List must not be empty.");
    let dim = first.dim();
    let grade = first.grade();
    assert!(elements.iter().all(|e| e.dim() == dim && e.grade() == grade));
    let columns: Vec<_> = elements.iter().map(|e| e.coeffs().column(0)).collect();
    Self::new(Matrix::from_columns(&columns), dim, grade)
  }
}
