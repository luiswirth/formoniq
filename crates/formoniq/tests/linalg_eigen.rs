//! [`sparse_shift_invert_eigen`]: every returned pair solves the pencil, a
//! degenerate cluster sitting at the shift (the harmonic-space case) is
//! resolved in full, and the spectrum of the 1D Dirichlet Laplacian is
//! reproduced from its closed form at a size no dense solver would run at.

use formoniq::linalg::eigen::sparse_shift_invert_eigen;
use simplicial::linalg::{CooMatrix, CsrMatrix, Matrix};

fn symmetric(n: usize, f: impl Fn(usize, usize) -> f64) -> Matrix {
  Matrix::from_fn(n, n, |i, j| f(i.min(j), i.max(j)))
}

fn csr(m: &Matrix) -> CsrMatrix {
  m.into()
}

/// Every returned pair solves the pencil: $A x = lambda B x$.
#[test]
fn pairs_solve_the_pencil() {
  let n = 6;
  let a = symmetric(n, |i, j| ((i * 7 + j * 3) % 11) as f64 - 5.0);
  // SPD: diagonally dominant with a positive diagonal.
  let b = symmetric(n, |i, j| if i == j { n as f64 } else { 0.3 });

  for nev in 1..=n {
    let (vals, vecs) = sparse_shift_invert_eigen(&csr(&a), &csr(&b), 0.0, nev).unwrap();
    for k in 0..vals.len() {
      let x = vecs.column(k).into_owned();
      let residual = (&a * &x - vals[k] * (&b * &x)).norm();
      assert!(residual < 1e-9, "nev={nev} k={k} residual={residual:e}");
    }
  }
}

/// A degenerate cluster of multiplicity `m` sitting exactly at the shift —
/// the harmonic-space case, `shift = 0` on a pencil where $A$ itself is
/// singular — is fully resolved (not collapsed to one direction), forcing
/// the shift-retry path every time.
#[test]
fn resolves_a_degenerate_cluster_at_the_shift() {
  let m = 3;
  let rest = 4;
  let n = m + rest;
  // B = I. A has an m-fold zero eigenvalue and `rest` nonzero ones, mixed by
  // a fixed change of basis so the cluster isn't axis-aligned with the seed.
  let b = Matrix::identity(n, n);
  let diag = Matrix::from_fn(n, n, |i, j| {
    if i != j || i < m {
      0.0
    } else {
      (i - m + 1) as f64
    }
  });
  let rot = Matrix::from_fn(n, n, |i, j| ((i * 5 + j * 3 + 1) % 7) as f64 - 3.0);
  let rot = rot.clone() + rot.transpose() + Matrix::identity(n, n) * (2.0 * n as f64);
  let a = &rot * &diag * &rot.transpose();
  let a = (&a + a.transpose()) * 0.5;

  let (vals, vecs) = sparse_shift_invert_eigen(&csr(&a), &csr(&b), 0.0, m).unwrap();
  assert_eq!(vals.len(), m);
  for &v in vals.iter() {
    assert!(v.abs() < 1e-6, "expected a near-zero eigenvalue, got {v}");
  }
  for k in 0..m {
    let x = vecs.column(k).into_owned();
    let residual = (&a * &x - vals[k] * &x).norm();
    assert!(residual < 1e-6, "k={k} residual={residual:e}");
  }
  // The m recovered eigenvectors are B-orthonormal, hence independent, hence
  // span the whole zero-eigenspace rather than repeating one direction.
  let gram = vecs.transpose() * &vecs;
  let dev = (&gram - Matrix::identity(m, m)).norm();
  assert!(
    dev < 1e-6,
    "recovered directions are not mutually independent: {gram}"
  );
}

/// The 1D Dirichlet Laplacian (tridiagonal, $B = I$), whose eigenvalues have
/// the closed form $lambda_j = 2 - 2 cos(j pi \/ (N + 1))$ — an oracle with
/// no dense EVD needed at any size, including $N$ in the low thousands,
/// where dense QZ has no business running.
#[test]
fn handles_a_large_sparse_pencil() {
  let n = 3000;
  let nev = 5;

  let mut coo = CooMatrix::new(n, n);
  for i in 0..n {
    coo.push(i, i, 2.0);
    if i + 1 < n {
      coo.push(i, i + 1, -1.0);
      coo.push(i + 1, i, -1.0);
    }
  }
  let a = CsrMatrix::from(&coo);
  let mut ident = CooMatrix::new(n, n);
  for i in 0..n {
    ident.push(i, i, 1.0);
  }
  let b = CsrMatrix::from(&ident);

  let (vals, vecs) = sparse_shift_invert_eigen(&a, &b, 0.0, nev).unwrap();
  assert_eq!(vals.len(), nev);
  for k in 0..nev {
    let x = vecs.column(k).into_owned();
    let residual = (&a * &x - vals[k] * (&b * &x)).norm();
    assert!(residual < 1e-6, "k={k} residual={residual:e}");

    let want = 2.0 - 2.0 * (((k + 1) as f64 * std::f64::consts::PI) / (n as f64 + 1.0)).cos();
    assert!(
      (vals[k] - want).abs() < 1e-6,
      "k={k}: got {} want {want}",
      vals[k]
    );
  }
}
