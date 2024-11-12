use faer::solvers::SpSolver;

#[derive(Default)]
pub struct SparseMatrix {
  nrows: usize,
  ncols: usize,
  triplets: Vec<(usize, usize, f64)>,
}

impl SparseMatrix {
  pub fn new(nrows: usize, ncols: usize) -> Self {
    Self::from_triplets(nrows, ncols, Vec::new())
  }

  pub fn from_triplets(nrows: usize, ncols: usize, triplets: Vec<(usize, usize, f64)>) -> Self {
    Self {
      nrows,
      ncols,
      triplets,
    }
  }

  pub fn nrows(&self) -> usize {
    self.nrows
  }
  pub fn ncols(&self) -> usize {
    self.ncols
  }

  pub fn ntriplets(&self) -> usize {
    self.triplets.len()
  }

  pub fn push(&mut self, r: usize, c: usize, v: f64) {
    self.triplets.push((r, c, v));
  }

  pub fn set_zero<F>(&mut self, predicate: F)
  where
    F: Fn(usize, usize) -> bool,
  {
    let mut i = 0;
    while i < self.triplets.len() {
      let triplet = self.triplets[i];
      let r = triplet.0;
      let c = triplet.1;
      if predicate(r, c) {
        self.triplets.swap_remove(i);
      } else {
        i += 1;
      }
    }
  }

  pub fn to_nalgebra_coo(&self) -> nas::CooMatrix<f64> {
    let rows = self.triplets.iter().map(|t| t.0).collect();
    let cols = self.triplets.iter().map(|t| t.1).collect();
    let vals = self.triplets.iter().map(|t| t.2).collect();
    nas::CooMatrix::try_from_triplets(self.nrows, self.ncols, rows, cols, vals).unwrap()
  }

  pub fn to_nalgebra_csc(&self) -> nas::CscMatrix<f64> {
    (&self.to_nalgebra_coo()).into()
  }

  pub fn to_nalgebra_dense(&self) -> na::DMatrix<f64> {
    (&self.to_nalgebra_coo()).into()
  }

  pub fn to_faer_csc(&self) -> faer::sparse::SparseColMat<usize, f64> {
    faer::sparse::SparseColMat::try_new_from_triplets(self.nrows, self.ncols, &self.triplets)
      .unwrap()
  }

  pub fn to_triplets(self) -> Vec<(usize, usize, f64)> {
    self.triplets
  }
}

type SparseMatrixFaer = faer::sparse::SparseColMat<usize, f64>;

pub fn nalgebra2faer(m: nas::CscMatrix<f64>) -> SparseMatrixFaer {
  let nrows = m.nrows();
  let ncols = m.ncols();
  let (col_ptrs, row_indices, values) = m.disassemble();

  let symbolic =
    faer::sparse::SymbolicSparseColMat::new_checked(nrows, ncols, col_ptrs, None, row_indices);
  faer::sparse::SparseColMat::new(symbolic, values)
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
  pub fn new(a: nas::CscMatrix<f64>) -> Self {
    let raw = nalgebra2faer(a).sp_lu().unwrap();
    Self { raw }
  }

  pub fn solve(&self, b: &na::DVector<f64>) -> na::DVector<f64> {
    let b = faer::col::from_slice(b.as_slice());
    na::DVector::from_vec(self.raw.solve(b).as_slice().to_vec())
  }
}

pub struct FaerCholesky {
  raw: faer::sparse::linalg::solvers::Cholesky<usize, f64>,
}
impl FaerCholesky {
  pub fn new(a: nas::CscMatrix<f64>) -> Self {
    let raw = nalgebra2faer(a).sp_cholesky(faer::Side::Upper).unwrap();
    Self { raw }
  }

  pub fn solve(&self, b: &na::DVector<f64>) -> na::DVector<f64> {
    let b = faer::col::from_slice(b.as_slice());
    na::DVector::from_vec(self.raw.solve(b).as_slice().to_vec())
  }
}
