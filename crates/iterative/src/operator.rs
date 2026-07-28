use crate::{CsrMatrix, LinearOperator, Vector};

/// The assembled sparse matrix is the archetypal operator: apply is one
/// sparse matrix-vector product.
impl LinearOperator for CsrMatrix {
  fn dim(&self) -> usize {
    debug_assert_eq!(self.nrows(), self.ncols(), "operator must be square");
    self.nrows()
  }
  fn apply(&self, x: &Vector) -> Vector {
    self * x
  }
}
