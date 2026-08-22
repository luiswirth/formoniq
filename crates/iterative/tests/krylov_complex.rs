//! The Hermitian convention over $CC$, where a misplaced conjugate is
//! visible.
//!
//! Conjugation is the identity on $RR$, so every law in `krylov.rs` passes on
//! an implementation whose inner product is bilinear rather than Hermitian,
//! or whose adjoint is a bare transpose. None of them can tell the
//! difference, which is exactly why this exists: the complex case is not an
//! extra feature being checked, it is the only place the convention is
//! observable. Real and complex being one implementation, this is the same
//! law swept over the field, not a second suite.

extern crate nalgebra as na;

mod common;

use common::{csr, dense_solve};
use iterative::krylov::{cg, minres};
use iterative::{CsrMatrix, Identity, InnerProductSpace, Report, StopCriterion, Vector, adjoint};
use na::{Complex, DMatrix};

type C = Complex<f64>;

fn c(re: f64, im: f64) -> C {
  Complex::new(re, im)
}

/// A Hermitian matrix with a prescribed (necessarily real) spectrum,
/// $A = Q Lambda Q^H$ with $Q$ a deterministic unitary factor.
///
/// Genuinely complex: $Q$ has a nonzero imaginary part, so $A^T != A^H$ and
/// the two conventions disagree on it.
fn hermitian_from_spectrum(eigs: &[f64]) -> DMatrix<C> {
  let n = eigs.len();
  let seed = DMatrix::from_fn(n, n, |i, j| {
    c(
      ((i * 7 + j * 13) % 11) as f64 - 5.0,
      ((i * 5 + j * 3) % 7) as f64 - 3.0,
    )
  });
  let q = seed.qr().q();
  let lambda = DMatrix::from_diagonal(&Vector::from_iterator(n, eigs.iter().map(|&e| c(e, 0.0))));
  &q * lambda * q.adjoint()
}

fn rhs(n: usize) -> Vector<C> {
  Vector::from_fn(n, |i, _| c((i as f64 + 1.0).sqrt(), (i as f64 - 2.0).cos()))
}

/// The inner product is Hermitian and the adjoint the conjugate transpose,
/// $angle.l A x, y angle.r = angle.l x, A^H y angle.r$, which is what makes
/// the Krylov recurrences orthogonal over $CC$: CG reaches the exact
/// solution of an $n times n$ Hermitian positive-definite system in at most
/// $n$ steps, and MINRES solves the Hermitian indefinite one CG cannot,
/// its Lanczos and rotation coefficients staying real throughout.
///
/// Under a bilinear inner product or a bare transpose the recurrences are no
/// longer conjugate-orthogonal, and neither method terminates. Swept over
/// orders with the degenerate $n = 0, 1$ included.
#[test]
fn the_krylov_methods_run_on_the_hermitian_inner_product() {
  let dense = DMatrix::from_fn(4, 3, |i, j| c(i as f64 - 1.0, 2.0 * j as f64 - 1.0));
  let dot = InnerProductSpace::dot;
  let (x, y) = (rhs(3), rhs(4));
  let ax = &dense * &x;
  let ahy = &DMatrix::from(&adjoint(&csr(&dense))) * &y;
  assert!((dot(&ax, &y) - dot(&x, &ahy)).norm() < 1e-12);

  for n in 0..=8 {
    let positive: Vec<f64> = (0..n).map(|k| 1.0 + k as f64).collect();
    let indefinite: Vec<f64> = (0..n)
      .map(|k| (k / 2 + 1) as f64 * if k % 2 == 0 { 1.0 } else { -1.0 })
      .collect();

    type Solver = fn(
      &CsrMatrix<C>,
      &Identity<Vector<C>>,
      &Vector<C>,
      StopCriterion<f64>,
    ) -> (Vector<C>, Report<f64>);

    for (eigs, solve) in [(positive, cg as Solver), (indefinite, minres as Solver)] {
      let dense = hermitian_from_spectrum(&eigs);
      let a = csr(&dense);
      let b = rhs(n);

      let stop = StopCriterion {
        rtol: 1e-11,
        max_iters: n.max(1),
      };
      let (x, report) = solve(&a, &Identity::new(n), &b, stop);
      assert!(report.converged, "n = {n} did not converge in {n} steps");
      if n > 0 {
        assert!((x - dense_solve(&dense, &b)).norm() < 1e-7, "n = {n}");
      }
    }
  }
}
