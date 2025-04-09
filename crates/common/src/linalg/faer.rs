use faer::linalg::solvers::Solve;

pub fn faervec2navec(faer: &faer::Mat<f64>) -> na::DVector<f64> {
  assert!(faer.ncols() == 1);
  na::DVector::from_iterator(faer.nrows(), faer.row_iter().map(|r| r[0]))
}

pub fn navec2faervec(na: &na::DVector<f64>) -> faer::Mat<f64> {
  let mut faer = faer::Mat::zeros(na.nrows(), 1);
  for (i, &v) in na.iter().enumerate() {
    faer[(i, 0)] = v;
  }
  faer
}

type SparseMatrixFaer = faer::sparse::SparseRowMat<usize, f64>;

pub fn nalgebra2faer(m: nas::CsrMatrix<f64>) -> SparseMatrixFaer {
  let nrows = m.nrows();
  let ncols = m.ncols();
  let (col_ptrs, row_indices, values) = m.disassemble();

  let symbolic =
    faer::sparse::SymbolicSparseRowMat::new_checked(nrows, ncols, col_ptrs, None, row_indices);
  faer::sparse::SparseRowMat::new(symbolic, values)
}

pub fn faer2nalgebra(m: SparseMatrixFaer) -> nas::CscMatrix<f64> {
  let (symbolic, values) = m.into_parts();
  let (nrows, ncols, col_ptrs, _, row_indices) = symbolic.into_parts();
  nas::CscMatrix::try_from_csc_data(nrows, ncols, col_ptrs, row_indices, values).unwrap()
}

pub struct FaerLu {
  raw: faer::sparse::linalg::solvers::Lu<usize, f64>,
}
impl FaerLu {
  pub fn new(a: nas::CsrMatrix<f64>) -> Self {
    let raw = nalgebra2faer(a).sp_lu().unwrap();
    Self { raw }
  }
  pub fn solve(&self, b: &na::DVector<f64>) -> na::DVector<f64> {
    let b = faer::Col::from_fn(b.nrows(), |i| b[i]);
    let x = self.raw.solve(b);
    na::DVector::from_iterator(x.nrows(), x.iter().copied())
  }
}

pub struct FaerCholesky {
  raw: faer::sparse::linalg::solvers::Llt<usize, f64>,
}
impl FaerCholesky {
  pub fn new(a: nas::CsrMatrix<f64>) -> Self {
    let raw = nalgebra2faer(a).sp_cholesky(faer::Side::Upper).unwrap();
    Self { raw }
  }

  pub fn solve(&self, b: &na::DVector<f64>) -> na::DVector<f64> {
    let b = faer::Col::from_fn(b.nrows(), |i| b[i]);
    let x = self.raw.solve(b);
    na::DVector::from_iterator(x.nrows(), x.iter().copied())
  }
}
