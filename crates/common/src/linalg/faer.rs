use faer::linalg::solvers::Solve;

use super::nalgebra::{CscMatrix, CsrMatrix, Matrix, Vector};

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
    let raw = nalgebra2faer(a).sp_lu().unwrap();
    Self { raw }
  }
  pub fn solve(&self, b: &Vector) -> Vector {
    let b = faer::Col::from_fn(b.nrows(), |i| b[i]);
    let x = self.raw.solve(b);
    Vector::from_iterator(x.nrows(), x.iter().copied())
  }
}

/// The generalized eigenproblem $A x = lambda B x$, solved for the
/// `neigenvalues` eigenpairs whose eigenvalue is nearest $0$.
///
/// A dense QZ factorization (`faer`'s `generalized_eigen`) of the pencil
/// $(A, B)$. It is deliberately the crudest correct method: it forms the whole
/// spectrum and keeps the few wanted pairs, so it is $O(n^3)$ time and $O(n^2)$
/// memory. That is right only while the meshes are small — the scaling
/// endpoint is a sparse shift-invert Lanczos, which this does not attempt.
///
/// QZ carries no definiteness assumption, so it is correct on the mixed
/// Hodge–Laplace pencil where $B$ is singular *and* indefinite: a null vector
/// of $B$ is returned as an infinite eigenvalue ($beta = 0$), which sorts to
/// the far end and is discarded by the nearest-to-$0$ selection. The pencils
/// here are symmetric with real spectrum; the imaginary parts QZ returns are
/// rounding noise and are dropped.
pub fn faer_ghiep(lhs: &CsrMatrix, rhs: &CsrMatrix, neigenvalues: usize) -> (Vector, Matrix) {
  let n = lhs.nrows();
  let lhs_dense = Matrix::from(lhs);
  let rhs_dense = Matrix::from(rhs);
  let a = faer::Mat::from_fn(n, n, |i, j| lhs_dense[(i, j)]);
  let b = faer::Mat::from_fn(n, n, |i, j| rhs_dense[(i, j)]);

  let gevd = a.generalized_eigen(&b).unwrap();
  let alpha = gevd.S_a().column_vector();
  let beta = gevd.S_b().column_vector();
  let vecs = gevd.U();

  // $lambda = alpha / beta$; an infinite eigenvalue ($beta = 0$) gets the key
  // $+oo$ so it sorts last and never enters the selection.
  let mut order: Vec<usize> = (0..n).collect();
  let key = |i: usize| {
    let lambda = *alpha.get(i) / *beta.get(i);
    if lambda.norm().is_finite() {
      lambda.norm()
    } else {
      f64::INFINITY
    }
  };
  order.sort_by(|&i, &j| key(i).total_cmp(&key(j)));
  order.truncate(neigenvalues);

  let eigenvals = Vector::from_iterator(
    order.len(),
    order.iter().map(|&i| (*alpha.get(i) / *beta.get(i)).re),
  );
  let eigenvecs = Matrix::from_fn(n, order.len(), |i, k| vecs.get(i, order[k]).re);
  (eigenvals, eigenvecs)
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

#[cfg(test)]
mod test {
  use super::{faer_ghiep, Matrix, Vector};

  fn symmetric(n: usize, f: impl Fn(usize, usize) -> f64) -> Matrix {
    Matrix::from_fn(n, n, |i, j| f(i.min(j), i.max(j)))
  }

  /// Every returned pair solves the pencil: $A x = lambda B x$.
  #[test]
  fn ghiep_pairs_solve_the_pencil() {
    let n = 6;
    let a = symmetric(n, |i, j| ((i * 7 + j * 3) % 11) as f64 - 5.0);
    // SPD: diagonally dominant with a positive diagonal.
    let b = symmetric(n, |i, j| if i == j { n as f64 } else { 0.3 });

    for nev in 1..=n {
      let (vals, vecs) = faer_ghiep(&(&a).into(), &(&b).into(), nev);
      for k in 0..vals.len() {
        let x = vecs.column(k).into_owned();
        let residual = (&a * &x - vals[k] * (&b * &x)).norm();
        assert!(residual < 1e-9, "nev={nev} k={k} residual={residual:e}");
      }
    }
  }

  /// With $B = I$ the pencil is the standard symmetric eigenproblem, so the
  /// eigenvalues nearest $0$ must match a dense symmetric EVD oracle.
  #[test]
  fn ghiep_matches_symmetric_evd_oracle() {
    let n = 7;
    let a = symmetric(n, |i, j| ((i * 5 + j * 2) % 13) as f64 - 6.0);
    let id = Matrix::identity(n, n);

    let mut oracle: Vec<f64> = a.clone().symmetric_eigenvalues().iter().copied().collect();
    oracle.sort_by(|x, y| x.abs().total_cmp(&y.abs()));

    for nev in 1..=n {
      let (vals, _) = faer_ghiep(&(&a).into(), &(&id).into(), nev);
      let mut got: Vec<f64> = vals.iter().copied().collect();
      let mut want = oracle[..nev].to_vec();
      got.sort_by(f64::total_cmp);
      want.sort_by(f64::total_cmp);
      for (g, w) in got.iter().zip(&want) {
        assert!((g - w).abs() < 1e-9, "nev={nev}: got {g} want {w}");
      }
    }
  }

  /// A singular, indefinite $B$ (the mixed-formulation regime): a null
  /// direction of $B$ is an infinite eigenvalue, excluded by the nearest-to-$0$
  /// selection, and the finite pairs it does return still solve the pencil.
  #[test]
  fn ghiep_excludes_infinite_eigenvalues() {
    let n = 5;
    let a = symmetric(n, |i, j| ((i * 3 + j * 7) % 11) as f64 - 5.0);
    // PSD but rank-deficient: the last coordinate carries no mass, so $B$ has a
    // one-dimensional kernel and the pencil has an eigenvalue at infinity.
    let b = Matrix::from_fn(n, n, |i, j| if i == j && i + 1 < n { 1.0 } else { 0.0 });

    let (vals, vecs) = faer_ghiep(&(&a).into(), &(&b).into(), n - 1);
    assert_eq!(vals.len(), n - 1);
    for k in 0..vals.len() {
      assert!(vals[k].is_finite());
      let x: Vector = vecs.column(k).into_owned();
      let residual = (&a * &x - vals[k] * (&b * &x)).norm();
      assert!(residual < 1e-9, "k={k} residual={residual:e}");
    }
  }
}
