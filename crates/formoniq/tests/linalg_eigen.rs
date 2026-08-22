//! [`sparse_shift_invert_eigen`]: every returned pair solves the pencil
//! $A x = lambda B x$ and the returned directions are $B$-orthonormal, over
//! the cases the solver is asked for.
//!
//! The cases are what makes the law bite: a generic indefinite pencil, a
//! degenerate cluster sitting exactly at the shift (the harmonic-space case,
//! which a solver may collapse to one direction), and the 1D Dirichlet
//! Laplacian at a size no dense solver would run at, whose closed-form
//! spectrum is the only external oracle available.

use formoniq::linalg::eigen::sparse_shift_invert_eigen;
use simplicial::linalg::{CooMatrix, CsrMatrix, Matrix};

fn symmetric(n: usize, f: impl Fn(usize, usize) -> f64) -> Matrix {
  Matrix::from_fn(n, n, |i, j| f(i.min(j), i.max(j)))
}

/// A pencil to solve, and the eigenvalues expected of it where a closed form
/// exists.
struct Pencil {
  name: String,
  a: CsrMatrix,
  b: CsrMatrix,
  nev: usize,
  oracle: Option<Vec<f64>>,
}

fn pencils() -> Vec<Pencil> {
  let mut pencils = Vec::new();

  // A generic symmetric pencil against an SPD (diagonally dominant) B, over
  // every count of requested pairs up to the full spectrum.
  let n = 6;
  let a = symmetric(n, |i, j| ((i * 7 + j * 3) % 11) as f64 - 5.0);
  let b = symmetric(n, |i, j| if i == j { n as f64 } else { 0.3 });
  for nev in 1..=n {
    pencils.push(Pencil {
      name: format!("generic pencil, nev = {nev}"),
      a: (&a).into(),
      b: (&b).into(),
      nev,
      oracle: None,
    });
  }

  // An m-fold zero eigenvalue sitting exactly at the shift, mixed by a fixed
  // change of basis so the cluster is not axis-aligned with the seed. This
  // forces the shift-retry path, and a solver that collapses the cluster
  // fails the orthonormality below rather than the residual.
  let (m, rest) = (3, 4);
  let n = m + rest;
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
  pencils.push(Pencil {
    name: "degenerate cluster at the shift".to_string(),
    a: (&a).into(),
    b: (&Matrix::identity(n, n)).into(),
    nev: m,
    oracle: Some(vec![0.0; m]),
  });

  // The 1D Dirichlet Laplacian, $lambda_j = 2 - 2 cos(j pi \/ (N + 1))$, at a
  // size where dense QZ has no business running.
  let n = 3000;
  let nev = 5;
  let mut coo = CooMatrix::new(n, n);
  let mut ident = CooMatrix::new(n, n);
  for i in 0..n {
    coo.push(i, i, 2.0);
    ident.push(i, i, 1.0);
    if i + 1 < n {
      coo.push(i, i + 1, -1.0);
      coo.push(i + 1, i, -1.0);
    }
  }
  let oracle = (1..=nev)
    .map(|j| 2.0 - 2.0 * ((j as f64 * std::f64::consts::PI) / (n as f64 + 1.0)).cos())
    .collect();
  pencils.push(Pencil {
    name: "1D Dirichlet Laplacian".to_string(),
    a: CsrMatrix::from(&coo),
    b: CsrMatrix::from(&ident),
    nev,
    oracle: Some(oracle),
  });

  pencils
}

/// $A x = lambda B x$ for every returned pair, with the returned directions
/// $B$-orthonormal, hence independent, hence spanning the eigenspace they
/// were asked for rather than repeating one direction.
#[test]
fn every_returned_pair_solves_the_pencil() {
  for Pencil {
    name,
    a,
    b,
    nev,
    oracle,
  } in pencils()
  {
    let (vals, vecs) = sparse_shift_invert_eigen(&a, &b, 0.0, nev).unwrap();

    for (k, &val) in vals.iter().enumerate() {
      let x = vecs.column(k).into_owned();
      let residual = (&a * &x - val * (&b * &x)).norm();
      assert!(residual < 1e-6, "{name}, k = {k}: residual {residual:e}");
    }

    let gram = vecs.transpose() * (&b * &vecs);
    let deviation = (&gram - Matrix::identity(vals.len(), vals.len())).norm();
    assert!(deviation < 1e-6, "{name}: directions are not B-orthonormal");

    if let Some(oracle) = oracle {
      assert_eq!(vals.len(), oracle.len(), "{name}");
      for (k, (&got, want)) in vals.iter().zip(oracle).enumerate() {
        assert!(
          (got - want).abs() < 1e-6,
          "{name}, k = {k}: {got} != {want}"
        );
      }
    }
  }
}
