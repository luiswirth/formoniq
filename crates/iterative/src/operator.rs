use crate::{CsrMatrix, LinearOperator, Vector};

/// The assembled sparse matrix is the archetypal operator: apply is one
/// sparse matrix-vector product.
impl LinearOperator for CsrMatrix {
  type Space = Vector;
  fn dim(&self) -> usize {
    debug_assert_eq!(self.nrows(), self.ncols(), "operator must be square");
    self.nrows()
  }
  fn apply(&self, x: &Vector) -> Vector {
    self * x
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::testutil::{csr, symmetric_from_spectrum};

  #[test]
  fn csr_matvec_matches_dense() {
    let dense = symmetric_from_spectrum(&[1.0, 2.0, 3.0, 4.0]);
    let a = csr(&dense);
    let x = Vector::from_column_slice(&[1.0, -2.0, 0.5, 3.0]);
    assert!((a.apply(&x) - &dense * &x).norm() < 1e-12);
    assert_eq!(LinearOperator::dim(&a), 4);
  }
}
