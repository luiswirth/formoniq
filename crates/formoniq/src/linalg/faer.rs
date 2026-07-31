use faer::linalg::solvers::Solve;

use simplicial::linalg::{CsrMatrix, Vector};

/// The scalar field a direct solve runs over, real or complex.
///
/// Two backends meet here and a scalar has to be a field for each: nalgebra
/// owns the assembled matrix, faer does the factorization. They agree on
/// `num_complex`, so this is one bound rather than a conversion between two
/// notions of a complex number.
pub trait Field: iterative::Field + faer::traits::ComplexField {}
impl<T: iterative::Field + faer::traits::ComplexField> Field for T {}

type SparseMatrixFaer<T> = faer::sparse::SparseRowMat<usize, T>;

fn nalgebra2faer<T: Field>(m: CsrMatrix<T>) -> SparseMatrixFaer<T> {
  let nrows = m.nrows();
  let ncols = m.ncols();
  let (col_ptrs, row_indices, values) = m.disassemble();

  let symbolic =
    faer::sparse::SymbolicSparseRowMat::new_checked(nrows, ncols, col_ptrs, None, row_indices);
  faer::sparse::SparseRowMat::new(symbolic, values)
}

fn faer2nalgebra<T: Field>(x: &faer::Col<T>) -> Vector<T> {
  Vector::from_iterator(x.nrows(), x.iter().copied())
}

fn nalgebra2faer_col<T: Field>(b: &Vector<T>) -> faer::Col<T> {
  faer::Col::from_fn(b.nrows(), |i| b[i])
}

/// Sparse LU factorization (faer): the direct solver for the symmetric
/// indefinite saddle-point systems of the mixed formulation.
///
/// The one factorization here that asks nothing of the matrix beyond
/// invertibility, which is what makes it the solver for a *complex-symmetric*
/// system, $A = A^T$ and not $A^H$. A lossy time-harmonic problem produces
/// exactly that, and no symmetric Krylov method applies to it.
pub struct FaerLu<T = f64> {
  raw: faer::sparse::linalg::solvers::Lu<usize, T>,
}
impl<T: Field> FaerLu<T> {
  pub fn new(a: CsrMatrix<T>) -> Self {
    Self::try_new(a).expect("sparse LU factorization failed")
  }
  /// Fallible variant of [`FaerLu::new`], for callers that can retry on a
  /// factorization failure (e.g. a shift landing exactly on an eigenvalue).
  pub fn try_new(a: CsrMatrix<T>) -> Option<Self> {
    let raw = nalgebra2faer(a).sp_lu().ok()?;
    Some(Self { raw })
  }
  pub fn solve(&self, b: &Vector<T>) -> Vector<T> {
    faer2nalgebra(&self.raw.solve(nalgebra2faer_col(b)))
  }
}

/// Sparse Cholesky factorization (faer): the direct solver for self-adjoint
/// positive-definite systems, $A = A^H$. Panics on an indefinite matrix; use
/// [`FaerLu`] there, and there too for a complex-symmetric one, which is not
/// self-adjoint.
pub struct FaerCholesky<T = f64> {
  raw: faer::sparse::linalg::solvers::Llt<usize, T>,
}
impl<T: Field> FaerCholesky<T> {
  pub fn new(a: CsrMatrix<T>) -> Self {
    Self::try_new(a).expect("sparse Cholesky factorization failed")
  }
  /// Fallible variant: `None` when the matrix is not positive definite (an
  /// indefinite mass on a Lorentzian geometry, say), for callers that fall back
  /// to an indefinite solver rather than panic.
  pub fn try_new(a: CsrMatrix<T>) -> Option<Self> {
    let raw = nalgebra2faer(a).sp_cholesky(faer::Side::Upper).ok()?;
    Some(Self { raw })
  }

  pub fn solve(&self, b: &Vector<T>) -> Vector<T> {
    faer2nalgebra(&self.raw.solve(nalgebra2faer_col(b)))
  }
}
