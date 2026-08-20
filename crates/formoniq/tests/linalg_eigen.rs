//! [`sparse_shift_invert_eigen`]: every returned pair solves the pencil,
//! matches a dense EVD oracle at $B = I$, the eigenvectors are
//! $B$-orthonormal, a null direction of $B$ is excluded, a degenerate
//! cluster at the shift is fully resolved, and a pencil with no finite
//! eigenvalue is reported rather than panicked on.

use formoniq::linalg::eigen::{EigenError, sparse_shift_invert_eigen};
use simplicial::linalg::{CooMatrix, CsrMatrix, Matrix, Vector};

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

/// With $B = I$ the pencil is the standard symmetric eigenproblem, so the
/// eigenvalues nearest $0$ must match a dense symmetric EVD oracle.
#[test]
fn matches_symmetric_evd_oracle() {
  let n = 7;
  let a = symmetric(n, |i, j| ((i * 5 + j * 2) % 13) as f64 - 6.0);
  let id = Matrix::identity(n, n);

  let mut oracle: Vec<f64> = a.clone().symmetric_eigenvalues().iter().copied().collect();
  oracle.sort_by(|x, y| x.abs().total_cmp(&y.abs()));

  for nev in 1..=n {
    let (vals, _) = sparse_shift_invert_eigen(&csr(&a), &csr(&id), 0.0, nev).unwrap();
    let mut got: Vec<f64> = vals.iter().copied().collect();
    let mut want = oracle[..nev].to_vec();
    got.sort_by(f64::total_cmp);
    want.sort_by(f64::total_cmp);
    for (g, w) in got.iter().zip(&want) {
      assert!((g - w).abs() < 1e-8, "nev={nev}: got {g} want {w}");
    }
  }
}

/// Eigenvectors are $B$-orthonormal.
#[test]
fn eigenvectors_are_b_orthonormal() {
  let n = 6;
  let a = symmetric(n, |i, j| if i == j { 2.0 * n as f64 } else { 0.5 });
  let b = symmetric(n, |i, j| if i == j { n as f64 } else { 0.3 });

  let (_, vecs) = sparse_shift_invert_eigen(&csr(&a), &csr(&b), 0.0, n).unwrap();
  for k in 0..vecs.ncols() {
    for l in 0..vecs.ncols() {
      let vk = vecs.column(k).into_owned();
      let vl = vecs.column(l).into_owned();
      let ip = vk.dot(&(&b * &vl));
      let want = if k == l { 1.0 } else { 0.0 };
      assert!((ip - want).abs() < 1e-8, "k={k} l={l} got {ip} want {want}");
    }
  }
}

/// A singular, indefinite-adjacent $B$ (the mixed-formulation regime): a
/// null direction of $B$ is never selected, and the finite pairs returned
/// still solve the pencil, even shifted away from $0$.
#[test]
fn excludes_the_null_space_of_b() {
  let n = 5;
  let a = symmetric(n, |i, j| ((i * 3 + j * 7) % 11) as f64 - 5.0);
  // PSD but rank-deficient: the last coordinate carries no mass.
  let b = Matrix::from_fn(n, n, |i, j| if i == j && i + 1 < n { 1.0 } else { 0.0 });

  let (vals, vecs) = sparse_shift_invert_eigen(&csr(&a), &csr(&b), 0.1, n - 1).unwrap();
  assert_eq!(vals.len(), n - 1);
  for k in 0..vals.len() {
    assert!(vals[k].is_finite());
    let x: Vector = vecs.column(k).into_owned();
    let residual = (&a * &x - vals[k] * (&b * &x)).norm();
    assert!(residual < 1e-8, "k={k} residual={residual:e}");
  }
}

/// The $1 times 1$ (and, via `k = 0`, empty) base case — the $0$-manifold's
/// Hodge-Laplace pencil.
#[test]
fn solves_the_scalar_pencil() {
  let a = Matrix::from_element(1, 1, 3.0);
  let b = Matrix::from_element(1, 1, 4.0);
  let (vals, vecs) = sparse_shift_invert_eigen(&csr(&a), &csr(&b), 0.0, 3).unwrap();
  assert_eq!(vals.len(), 1);
  assert!((vals[0] - 3.0 / 4.0).abs() < 1e-9);
  let x: Vector = vecs.column(0).into_owned();
  assert!(
    (x.dot(&(&b * &x)) - 1.0).abs() < 1e-9,
    "eigenvector is B-normalized"
  );
  assert!((&a * &x - vals[0] * (&b * &x)).norm() < 1e-9);

  let (vals0, vecs0) = sparse_shift_invert_eigen(&csr(&a), &csr(&b), 0.0, 0).unwrap();
  assert_eq!(vals0.len(), 0);
  assert_eq!(vecs0.ncols(), 0);
}

/// A pencil with no finite eigenvalue at all ($B = 0$) is reported, not
/// panicked on.
#[test]
fn reports_a_pencil_without_finite_eigenvalues() {
  let a = Matrix::identity(4, 4);
  let b = Matrix::zeros(4, 4);
  assert_eq!(
    sparse_shift_invert_eigen(&csr(&a), &csr(&b), 0.0, 2),
    Err(EigenError::NoFiniteEigenvalue)
  );
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
