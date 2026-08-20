//! The same laws over $CC$, where a misplaced conjugate is visible.
//!
//! Conjugation is the identity on $RR$, so every law in `krylov.rs` passes on
//! an implementation whose inner product is bilinear rather than Hermitian,
//! or whose adjoint is a bare transpose. None of them can tell the
//! difference, which is exactly why these exist: the complex case is not an
//! extra feature being checked, it is the only place the convention is
//! observable.

extern crate nalgebra as na;

mod common;

use common::{csr, dense_solve, symmetric_from_spectrum};
use iterative::krylov::cg;
use iterative::{Identity, InnerProductSpace, Jacobi, StopCriterion, Vector, adjoint};
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

/// The inner product is sesquilinear, conjugate-linear in its first argument:
/// $angle.l i x, y angle.r = -i angle.l x, y angle.r$ and
/// $angle.l x, i y angle.r = i angle.l x, y angle.r$.
///
/// The two halves must be checked separately. A bilinear `dot` satisfies
/// neither, and a `dot` conjugating the *other* argument satisfies both with
/// the signs exchanged, which is the mistake a single-sided test misses.
#[test]
fn the_inner_product_is_conjugate_linear_in_its_first_argument() {
  // Spelled through the trait, never as `x.dot(&y)`: nalgebra's inherent
  // `dot` is the *bilinear* product and wins method resolution on a concrete
  // vector, so the shorthand would test nalgebra rather than the trait. The
  // generic code cannot make this mistake, having no inherent method to find.
  let dot = InnerProductSpace::dot;
  let (x, y) = (
    rhs(5),
    Vector::from_fn(5, |i, _| c((i as f64).sin(), 1.0 - i as f64)),
  );
  let xy = dot(&x, &y);
  let i = c(0.0, 1.0);

  assert!((dot(&(x.clone() * i), &y) - (-i) * xy).norm() < 1e-12);
  assert!((dot(&x, &(y.clone() * i)) - i * xy).norm() < 1e-12);
  // And it is positive definite, so the induced norm is real.
  assert!(dot(&x, &x).im.abs() < 1e-12 && dot(&x, &x).re > 0.0);
}

/// The adjoint is the conjugate transpose, $(A^H)_(i j) = overline(A_(j i))$,
/// and it is what makes $angle.l A x, y angle.r = angle.l x, A^H y angle.r$.
/// The bare transpose satisfies neither over $CC$.
#[test]
fn the_adjoint_is_the_conjugate_transpose() {
  let dense = DMatrix::from_fn(4, 3, |i, j| c(i as f64 - 1.0, 2.0 * j as f64 - 1.0));
  let a = csr(&dense);
  let dot = InnerProductSpace::dot;
  let (x, y) = (rhs(3), rhs(4));
  let ax = &dense * &x;
  let ahy = &DMatrix::from(&adjoint(&a)) * &y;
  assert!((dot(&ax, &y) - dot(&x, &ahy)).norm() < 1e-12);
}

/// CG's defining theorem over $CC$: on an $n times n$ Hermitian
/// positive-definite system it reaches the exact solution in at most $n$
/// steps. Swept over orders with the degenerate $n = 0, 1$ included.
///
/// This is the test a bilinear inner product fails: the recurrence is no
/// longer conjugate-orthogonal, so it neither terminates nor converges.
#[test]
fn cg_terminates_on_a_hermitian_positive_definite_system() {
  for n in 0..=8 {
    let eigs: Vec<f64> = (0..n).map(|k| 1.0 + k as f64).collect();
    let dense = hermitian_from_spectrum(&eigs);
    let a = csr(&dense);
    let b = rhs(n);

    let stop = StopCriterion {
      rtol: 1e-10,
      max_iters: n.max(1),
    };
    let (x, report) = cg(&a, &Identity::new(n), &b, stop);
    assert!(report.converged, "n = {n} did not converge in {n} steps");
    if n > 0 {
      assert!((x - dense_solve(&dense, &b)).norm() < 1e-7, "n = {n}");
    }
  }
}

/// MINRES solves a Hermitian *indefinite* complex system, the case CG cannot,
/// reproducing the direct solve. Its Lanczos and rotation coefficients are
/// real throughout, which is what self-adjointness buys.
#[test]
fn minres_solves_a_hermitian_indefinite_system() {
  use iterative::krylov::minres;
  for n in 0..=8 {
    let eigs: Vec<f64> = (0..n)
      .map(|k| (k / 2 + 1) as f64 * if k % 2 == 0 { 1.0 } else { -1.0 })
      .collect();
    let dense = hermitian_from_spectrum(&eigs);
    let a = csr(&dense);
    let b = rhs(n);

    let (x, report) = minres(&a, &Identity::new(n), &b, StopCriterion::rtol(1e-11));
    assert!(report.converged, "n = {n} did not converge");
    if n > 0 {
      assert!((x - dense_solve(&dense, &b)).norm() < 1e-7, "n = {n}");
    }
  }
}

/// Preconditioning a complex system changes the path, never the fixed point.
/// Jacobi reads a Hermitian operator's diagonal, which is real.
#[test]
fn preconditioning_preserves_the_complex_solution() {
  let dense = hermitian_from_spectrum(&[1.0, 2.0, 3.5, 6.0, 11.0, 14.0]);
  let a = csr(&dense);
  let b = rhs(6);
  let stop = StopCriterion::rtol(1e-12);

  let (x_plain, _) = cg(&a, &Identity::new(6), &b, stop);
  let (x_jacobi, _) = cg(&a, &Jacobi::new(&a), &b, stop);
  assert!((&x_plain - dense_solve(&dense, &b)).norm() < 1e-9);
  assert!((x_plain - x_jacobi).norm() < 1e-9);
}

/// A real system embedded in $CC$ has the real solution: extension of scalars
/// commutes with the solve, so the complex instantiation is a generalization
/// of the real one rather than a parallel implementation of it.
#[test]
fn a_real_system_solved_over_the_complexes_stays_real() {
  let dense = symmetric_from_spectrum(&[1.0, 2.0, 4.0, 7.0, 9.0]);
  let b = Vector::from_fn(5, |i, _| (i as f64 + 1.0).ln());
  let stop = StopCriterion::rtol(1e-12);
  let (x_real, _) = cg(&csr(&dense), &Identity::new(5), &b, stop);

  let dense_c = dense.map(|v| c(v, 0.0));
  let (x_complex, _) = cg(
    &csr(&dense_c),
    &Identity::new(5),
    &b.map(|v| c(v, 0.0)),
    StopCriterion::rtol(1e-12),
  );
  assert!(x_complex.iter().all(|z| z.im.abs() < 1e-12));
  assert!((x_complex.map(|z| z.re) - x_real).norm() < 1e-12);
}
