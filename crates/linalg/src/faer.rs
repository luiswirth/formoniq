use faer::linalg::solvers::Solve;

use super::nalgebra::{CscMatrix, CsrMatrix, Vector};

pub fn faervec2navec(faer: &faer::Mat<f64>) -> Vector {
  assert_eq!(faer.ncols(), 1);
  Vector::from_iterator(faer.nrows(), faer.row_iter().map(|r| r[0]))
}

pub fn navec2faervec(na: &Vector) -> faer::Mat<f64> {
  let mut faer = faer::Mat::zeros(na.nrows(), 1);
  for (i, &v) in na.iter().enumerate() {
    faer[(i, 0)] = v;
  }
  faer
}

type SparseMatrixFaer = faer::sparse::SparseRowMat<usize, f64>;

pub fn nalgebra2faer(m: CsrMatrix) -> SparseMatrixFaer {
  let nrows = m.nrows();
  let ncols = m.ncols();
  let (col_ptrs, row_indices, values) = m.disassemble();

  let symbolic =
    faer::sparse::SymbolicSparseRowMat::new_checked(nrows, ncols, col_ptrs, None, row_indices);
  faer::sparse::SparseRowMat::new(symbolic, values)
}

pub fn faer2nalgebra(m: SparseMatrixFaer) -> CscMatrix {
  let (symbolic, values) = m.into_parts();
  let (nrows, ncols, col_ptrs, _, row_indices) = symbolic.into_parts();
  CscMatrix::try_from_csc_data(nrows, ncols, col_ptrs, row_indices, values).unwrap()
}

pub struct FaerLu {
  raw: faer::sparse::linalg::solvers::Lu<usize, f64>,
}
impl FaerLu {
  pub fn new(a: CsrMatrix) -> Self {
    Self::try_new(a).expect("sparse LU factorization failed")
  }
  /// Fallible variant of [`FaerLu::new`], for callers that can retry on a
  /// factorization failure (e.g. a shift landing exactly on an eigenvalue).
  pub fn try_new(a: CsrMatrix) -> Option<Self> {
    let raw = nalgebra2faer(a).sp_lu().ok()?;
    Some(Self { raw })
  }
  pub fn solve(&self, b: &Vector) -> Vector {
    let b = faer::Col::from_fn(b.nrows(), |i| b[i]);
    let x = self.raw.solve(b);
    Vector::from_iterator(x.nrows(), x.iter().copied())
  }
}

pub struct FaerCholesky {
  raw: faer::sparse::linalg::solvers::Llt<usize, f64>,
}
impl FaerCholesky {
  pub fn new(a: CsrMatrix) -> Self {
    let raw = nalgebra2faer(a).sp_cholesky(faer::Side::Upper).unwrap();
    Self { raw }
  }

  pub fn solve(&self, b: &Vector) -> Vector {
    let b = faer::Col::from_fn(b.nrows(), |i| b[i]);
    let x = self.raw.solve(b);
    Vector::from_iterator(x.nrows(), x.iter().copied())
  }
}
