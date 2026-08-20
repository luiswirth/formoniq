//! [`LinearOperator`] for a [`CsrMatrix`]: applying it is the sparse
//! matrix-vector product.

mod common;

use common::{csr, symmetric_from_spectrum};
use iterative::{LinearOperator, Vector};

#[test]
fn csr_matvec_matches_dense() {
  let dense = symmetric_from_spectrum(&[1.0, 2.0, 3.0, 4.0]);
  let a = csr(&dense);
  let x = Vector::from_column_slice(&[1.0, -2.0, 0.5, 3.0]);
  assert!((a.apply(&x) - &dense * &x).norm() < 1e-12);
  assert_eq!(LinearOperator::dim(&a), 4);
}
