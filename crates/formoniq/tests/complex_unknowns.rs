//! Complex unknowns over a real operator: the seam, end to end.
//!
//! The geometry is real, the Whitney forms are real and the metric is real, so
//! every element matrix and every assembled operator of a real-coefficient
//! problem is real *whatever field its solution lives in*. A complex problem
//! therefore extends the operator it already has rather than assembling a
//! second one, and the whole complex capability of the engine is that seam plus
//! a solver that runs over the extended field.
//!
//! These are the laws of that seam. They are the ones a wrong conjugate breaks,
//! and none of them can be stated over the reals, where conjugation is the
//! identity.

use formoniq::{
  linalg::{
    DirectInverse, bilinear_form_sparse,
    faer::{FaerCholesky, FaerLu},
    quadratic_form_sparse, sesquilinear_form_sparse,
  },
  whitney_complex::{HilbertComplex, WhitneyComplex},
};
use iterative::{Identity, StopCriterion, krylov::cg};
use na::Complex;
use regge::mesher::cartesian::CartesianGrid;
use simplicial::{
  Dim,
  linalg::{CsrMatrix, CsrMatrixExt, Vector},
  topology::chain::Cochain,
};

extern crate nalgebra as na;

type C = Complex<f64>;

fn re(x: f64) -> C {
  Complex::new(x, 0.0)
}

/// The real Whitney mass and the Hodge-Laplace term at one grade of a small
/// unit grid: an operator assembled exactly as any real problem assembles it.
fn real_operators(dim: Dim, grade: usize) -> (CsrMatrix, CsrMatrix) {
  let (topology, coords) = CartesianGrid::new_unit(dim, 3).triangulate();
  let lengths = coords.to_edge_lengths_sq(&topology);
  let whitney = WhitneyComplex::new(&topology, &lengths);
  (
    CsrMatrix::from(&whitney.mass(grade)),
    CsrMatrix::from(&whitney.codif_dif(grade)),
  )
}

fn probe(n: usize, seed: usize) -> Vector {
  Vector::from_fn(n, |i, _| (((i + seed) % 7) as f64 - 3.0) * 0.5)
}

/// Extension of scalars commutes with the solve: over a *real* operator the
/// complex solution is the pair of real solutions, $M(x_r + i x_i) = b_r + i
/// b_i$ iff $M x_r = b_r$ and $M x_i = b_i$.
///
/// This is the law that makes the seam a seam rather than a second engine. It
/// also pins the complex Krylov path against the real direct one: a bilinear
/// inner product would not reach either half.
#[test]
fn a_complex_solve_of_a_real_operator_is_two_real_solves() {
  for dim in (1..=3).map(Dim::from) {
    for grade in 0..=dim.index() {
      let (mass, _) = real_operators(dim, grade);
      let n = mass.nrows();
      if n == 0 {
        continue;
      }
      let (br, bi) = (probe(n, 1), probe(n, 4));

      let real_re = FaerCholesky::new(mass.clone()).solve(&br);
      let real_im = FaerCholesky::new(mass.clone()).solve(&bi);

      // The seam: the same operator, read over CC.
      let mass_c = mass.extend_scalars(|&v| re(v));
      let b_c = Vector::from_fn(n, |i, _| Complex::new(br[i], bi[i]));
      let (x_c, report) = cg(&mass_c, &Identity::new(n), &b_c, StopCriterion::rtol(1e-12));
      assert!(report.converged, "dim={dim} grade={grade} did not converge");

      let got_re = Vector::from_fn(n, |i, _| x_c[i].re);
      let got_im = Vector::from_fn(n, |i, _| x_c[i].im);
      assert!(
        (got_re - &real_re).norm() < 1e-9 && (got_im - &real_im).norm() < 1e-9,
        "dim={dim} grade={grade}: the complex solve did not split"
      );
    }
  }
}

/// A cochain extends along the same ring map as its operator, and the two
/// commute with applying it: $phi(M c) = phi(M) phi(c)$.
///
/// The operator side and the coefficient side of one pairing, which is why
/// `CsrMatrix::extend_scalars` and `FreeModule::extend_scalars` are the same
/// word.
#[test]
fn the_operator_and_the_cochain_extend_together() {
  let dim = Dim::new(2);
  for grade in 0..=dim.index() {
    let (mass, _) = real_operators(dim, grade);
    let n = mass.nrows();
    let c: Cochain = Cochain::new(grade, probe(n, 2));

    let applied_then_extended = (&mass * c.coeffs()).map(re);
    let extended_then_applied =
      &mass.extend_scalars(|&v| re(v)) * &c.extend_scalars(|&v| re(v)).coeffs().clone();
    assert!((applied_then_extended - extended_then_applied).norm() < 1e-12);
  }
}

/// The time-harmonic shape: $S = K - (omega^2 + i omega sigma) M$ with real
/// $K$ and $M$, which is complex *symmetric*, $S = S^T$, and not self-adjoint.
///
/// The distinction is the point. A symmetric Krylov method requires $S = S^H$
/// and this is not that, so the direct LU is what solves it. The test asserts
/// the non-Hermiticity as well as the residual: without it the case would be
/// silently Hermitian and would say nothing about the one that is not.
#[test]
fn a_lossy_time_harmonic_system_is_complex_symmetric_and_solved_directly() {
  let dim = Dim::new(2);
  for grade in 0..=dim.index() {
    let (mass, stiff) = real_operators(dim, grade);
    let n = mass.nrows();
    if n == 0 {
      continue;
    }
    let (omega, sigma) = (2.0, 0.35);
    let coefficient = Complex::new(omega * omega, omega * sigma);

    let system = &stiff.extend_scalars(|&v| re(v)) - &mass.extend_scalars(|&v| re(v) * coefficient);

    // Symmetric but not self-adjoint, so this is genuinely the case CG and
    // MINRES do not cover.
    let transpose = system.transpose();
    let adjoint = iterative::adjoint(&system);
    assert!(
      (&system - &transpose)
        .values()
        .iter()
        .all(|v| v.norm() < 1e-12),
      "grade={grade}: the system should be complex symmetric"
    );
    assert!(
      (&system - &adjoint)
        .values()
        .iter()
        .any(|v| v.norm() > 1e-6),
      "grade={grade}: the system is self-adjoint, so it does not exercise the \
       complex-symmetric case at all"
    );

    let b = Vector::from_fn(n, |i, _| Complex::new(probe(n, 3)[i], probe(n, 6)[i]));
    let x = FaerLu::new(system.clone()).solve(&b);
    assert!(
      (&system * &x - &b).norm() < 1e-8 * b.norm(),
      "grade={grade}: the direct solve did not reproduce the right-hand side"
    );
  }
}

/// The energy of a complex cochain is real and positive, and the sesquilinear
/// form is the one that gives it: $c^H M c = norm(c_r)_M^2 + norm(c_i)_M^2$.
///
/// The bilinear form $c^T M c$ is a different number, complex in general, and
/// the two agree over $RR$. That is the whole reason they are two functions.
#[test]
fn the_energy_of_a_complex_cochain_is_the_sesquilinear_form() {
  let dim = Dim::new(2);
  for grade in 0..=dim.index() {
    let (mass, _) = real_operators(dim, grade);
    let n = mass.nrows();
    if n == 0 {
      continue;
    }
    let (cr, ci) = (probe(n, 1), probe(n, 5));
    let mass_c = mass.extend_scalars(|&v| re(v));
    let c = Vector::from_fn(n, |i, _| Complex::new(cr[i], ci[i]));

    let energy = quadratic_form_sparse(&mass_c, &c);
    let split = quadratic_form_sparse(&mass, &cr) + quadratic_form_sparse(&mass, &ci);
    assert!((energy - split).abs() < 1e-10, "grade={grade}");
    assert!(energy > 0.0, "grade={grade}: a mass energy is positive");

    // And it is genuinely not the bilinear form, which the reals cannot show.
    let bilinear = bilinear_form_sparse(&mass_c, &c, &c);
    assert!(
      (bilinear - Complex::new(energy, 0.0)).norm() > 1e-6,
      "grade={grade}: the two forms agree, so the sweep says nothing"
    );
    assert!((sesquilinear_form_sparse(&mass_c, &c, &c) - Complex::new(energy, 0.0)).norm() < 1e-10);
  }
}

/// The direct SPD inverse runs over the extended field and stays the exact
/// inverse there: `DirectInverse` is a factorization, not a real-only path.
#[test]
fn the_direct_inverse_runs_over_the_complexes() {
  let dim = Dim::new(2);
  let (mass, _) = real_operators(dim, 1);
  let n = mass.nrows();
  let mass_c = mass.extend_scalars(|&v| re(v));
  let inverse = DirectInverse::new(mass_c.clone());

  let b = Vector::from_fn(n, |i, _| Complex::new(probe(n, 2)[i], probe(n, 7)[i]));
  let x = iterative::ApproxInverse::apply(&inverse, &b);
  assert!((&mass_c * &x - &b).norm() < 1e-9 * b.norm());
}
