use crate::{CsrMatrix, Field, LinearOperator, Vector};

/// The assembled sparse matrix is the archetypal operator: apply is one
/// sparse matrix-vector product.
impl<T: Field> LinearOperator for CsrMatrix<T> {
  type Space = Vector<T>;
  fn dim(&self) -> usize {
    debug_assert_eq!(self.nrows(), self.ncols(), "operator must be square");
    self.nrows()
  }
  fn apply(&self, x: &Vector<T>) -> Vector<T> {
    self * x
  }
}
