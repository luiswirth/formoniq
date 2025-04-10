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
    let mut iter = iter.into_iter();
    let first = iter.next().unwrap();
    let dim = first.dim();
    let grade = first.grade();
    let mut coeffs = Matrix::zeros(first.coeffs.len(), 1);
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
