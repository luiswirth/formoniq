//! Complex unknowns over a real operator: the seam, end to end.
//!
//! The geometry is real, the Whitney forms are real and the metric is real, so
//! every element matrix and every assembled operator of a real-coefficient
//! problem is real *whatever field its solution lives in*. A complex problem
//! therefore extends the operator it already has rather than assembling a
//! second one, and the whole complex capability of the engine is that seam plus
//! a solver that runs over the extended field.
//!
//! This is the law of that seam, and it cannot be stated over the reals, where
//! conjugation is the identity.

use formoniq::{
  linalg::faer::FaerCholesky,
  whitney_complex::{HilbertComplex, WhitneyComplex},
};
use iterative::{Identity, StopCriterion, krylov::cg};
use na::Complex;
use regge::mesher::cartesian::CartesianGrid;
use simplicial::{
  Dim,
  linalg::{CsrMatrix, CsrMatrixExt, Vector},
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
  (whitney.mass(grade), whitney.dif_both(grade + 1))
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
