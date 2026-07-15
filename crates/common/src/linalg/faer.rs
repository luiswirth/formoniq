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

  // faer's dense QZ mis-sizes its workspace for $n <= 1$, but the generalized
  // eigenproblem is trivial at that size: the $1 times 1$ pencil $(a, b)$ has the
  // single eigenvalue $a / b$ with $B$-normalized eigenvector $1 / sqrt(b)$, and
  // the empty pencil has none. This is the base case that lets the
  // $0$-dimensional manifold (a point, on which the Hodge Laplacian is the zero
  // operator) solve rather than crash.
  if n <= 1 {
    let count = neigenvalues.min(n);
    let eigenvals = Vector::from_iterator(
      count,
      (0..count).map(|i| lhs_dense[(i, i)] / rhs_dense[(i, i)]),
    );
    let eigenvecs = Matrix::from_fn(n, count, |i, _| 1.0 / rhs_dense[(i, i)].sqrt());
    return (eigenvals, eigenvecs);
  }

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

/// The generalized symmetric-definite eigenproblem $A x = lambda B x$, with $A$
/// symmetric and $B$ symmetric *positive definite*, solved for the
/// `neigenvalues` algebraically smallest pairs. Eigenvalues are returned in
/// ascending order; eigenvectors are $B$-orthonormal.
///
/// Where [`faer_ghiep`] carries no definiteness assumption and pays for it with
/// a nonsymmetric QZ, this exploits the structure of the elliptic pencil (the
/// FE stiffness against an SPD mass matrix). A Cholesky factor $B = L L^top$
/// reduces the pencil to the standard self-adjoint problem $C y = lambda y$ with
/// $C = L^(-1) A L^(-top)$, whose spectrum is real, whose solver is a symmetric
/// tridiagonal QR — an order of magnitude cheaper than QZ and parallel — and
/// whose eigenvectors back-transform as $x = L^(-top) y$.
///
/// Still dense: it forms the whole spectrum and keeps the smallest
/// `neigenvalues`, so $O(n^3)$ time and $O(n^2)$ memory, right only while the
/// meshes are small. The scaling endpoint is a sparse shift-invert Lanczos,
/// which this does not attempt. $B$ *must* be positive definite; on the
/// singular, indefinite pencil of the mixed formulation, use [`faer_ghiep`].
pub fn faer_gsdiep(lhs: &CsrMatrix, rhs: &CsrMatrix, neigenvalues: usize) -> (Vector, Matrix) {
  let n = lhs.nrows();
  let lhs_dense = Matrix::from(lhs);
  let rhs_dense = Matrix::from(rhs);
  let count = neigenvalues.min(n);

  // The $1 times 1$ (and empty) base case, mirroring faer_ghiep: the pencil
  // $(a, b)$ has the single eigenvalue $a / b$ with $B$-normalized eigenvector
  // $1 / sqrt(b)$. This keeps the $0$-manifold total rather than tripping the
  // Cholesky/EVD on a degenerate size.
  if n <= 1 {
    let eigenvals = Vector::from_iterator(
      count,
      (0..count).map(|i| lhs_dense[(i, i)] / rhs_dense[(i, i)]),
    );
    let eigenvecs = Matrix::from_fn(n, count, |i, _| 1.0 / rhs_dense[(i, i)].sqrt());
    return (eigenvals, eigenvecs);
  }

  let a = faer::Mat::from_fn(n, n, |i, j| lhs_dense[(i, j)]);
  let b = faer::Mat::from_fn(n, n, |i, j| rhs_dense[(i, j)]);

  let l = b
    .llt(faer::Side::Lower)
    .expect("B is not positive definite; use faer_ghiep on an indefinite pencil")
    .L()
    .to_owned();

  // $C = L^(-1) A L^(-top)$ via two lower-triangular solves. With $A$ symmetric,
  // $W = L^(-1) A$ has $W^top = A L^(-top)$, so $L^(-1) W^top = C$ is symmetric;
  // the EVD reads only its lower triangle, absorbing the rounding asymmetry.
  let mut w = a;
  l.solve_lower_triangular_in_place(w.as_mut());
  let mut c = w.transpose().to_owned();
  l.solve_lower_triangular_in_place(c.as_mut());

  let eig = c.self_adjoint_eigen(faer::Side::Lower).unwrap();
  let vals = eig.S().column_vector();
  let y = eig.U();

  // The smallest `count` are the leading columns; back-transform $x = L^(-top) y$.
  let mut x = faer::Mat::from_fn(n, count, |i, k| *y.get(i, k));
  l.transpose().solve_upper_triangular_in_place(x.as_mut());

  let eigenvals = Vector::from_iterator(count, (0..count).map(|i| *vals.get(i)));
  let eigenvecs = Matrix::from_fn(n, count, |i, k| *x.get(i, k));
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
  use super::{faer_ghiep, faer_gsdiep, Matrix, Vector};

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

  /// The $1 times 1$ base case (the $0$-manifold's Hodge-Laplace pencil), where
  /// faer's dense QZ mis-sizes its workspace: the lone eigenvalue is $a / b$ with
  /// a $B$-normalized eigenvector, and it still solves the pencil.
  #[test]
  fn ghiep_solves_the_scalar_pencil() {
    let a = Matrix::from_element(1, 1, 3.0);
    let b = Matrix::from_element(1, 1, 4.0);
    let (vals, vecs) = faer_ghiep(&(&a).into(), &(&b).into(), 3);
    assert_eq!(vals.len(), 1);
    assert!((vals[0] - 3.0 / 4.0).abs() < 1e-12);
    let x: Vector = vecs.column(0).into_owned();
    assert!(
      (x.dot(&(&b * &x)) - 1.0).abs() < 1e-12,
      "eigenvector is B-normalized"
    );
    assert!((&a * &x - vals[0] * (&b * &x)).norm() < 1e-12);
  }

  /// The symmetric-definite solver agrees with the general QZ on a definite
  /// pencil — same smallest eigenvalues — and every returned pair solves
  /// $A x = lambda B x$ with a $B$-orthonormal eigenvector.
  #[test]
  fn gsdiep_matches_ghiep_on_definite_pencil() {
    let n = 6;
    // The elliptic regime gsdiep is for: an SPD $A$ (positive spectrum), where
    // "algebraically smallest" and "smallest magnitude" coincide, so gsdiep's
    // ascending selection matches ghiep's magnitude selection. On an indefinite
    // $A$ the two conventions legitimately diverge and are not meant to agree.
    let a = symmetric(n, |i, j| if i == j { 2.0 * n as f64 } else { 0.5 });
    // SPD: diagonally dominant with a positive diagonal.
    let b = symmetric(n, |i, j| if i == j { n as f64 } else { 0.3 });

    for nev in 1..=n {
      let (vals, vecs) = faer_gsdiep(&(&a).into(), &(&b).into(), nev);
      let (want, _) = faer_ghiep(&(&a).into(), &(&b).into(), nev);

      let mut got: Vec<f64> = vals.iter().copied().collect();
      let mut want: Vec<f64> = want.iter().copied().collect();
      got.sort_by(f64::total_cmp);
      want.sort_by(f64::total_cmp);
      for (g, w) in got.iter().zip(&want) {
        assert!((g - w).abs() < 1e-9, "nev={nev}: got {g} want {w}");
      }

      for k in 0..vals.len() {
        let x = vecs.column(k).into_owned();
        let residual = (&a * &x - vals[k] * (&b * &x)).norm();
        assert!(residual < 1e-9, "nev={nev} k={k} residual={residual:e}");
        let bnorm = x.dot(&(&b * &x));
        assert!(
          (bnorm - 1.0).abs() < 1e-9,
          "nev={nev} k={k} not B-normalized"
        );
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
