use faer::linalg::solvers::Solve;

use simplicial::linalg::{CsrMatrix, Vector};

type SparseMatrixFaer = faer::sparse::SparseRowMat<usize, f64>;

fn nalgebra2faer(m: CsrMatrix) -> SparseMatrixFaer {
  let nrows = m.nrows();
  let ncols = m.ncols();
  let (col_ptrs, row_indices, values) = m.disassemble();

  let symbolic =
    faer::sparse::SymbolicSparseRowMat::new_checked(nrows, ncols, col_ptrs, None, row_indices);
  faer::sparse::SparseRowMat::new(symbolic, values)
}

/// Sparse LU factorization (faer): the direct solver for the symmetric
/// indefinite saddle-point systems of the mixed formulation.
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

/// Sparse Cholesky factorization (faer): the direct solver for symmetric
/// positive-definite systems. Panics on an indefinite matrix; use [`FaerLu`]
/// there.
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
