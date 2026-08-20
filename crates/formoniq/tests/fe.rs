//! The $L^2$ projection onto the Whitney space: $P_h compose W = id$, the
//! error vanishes exactly on that space, and the mass solve matches between
//! CG and a direct Cholesky factorization.

use approx::assert_relative_eq;
use derham::Cochain;
use derham::interpolate::interpolant::WhitneyInterpolant;
use derham::section::CoordFieldExt;
use formoniq::fe::{fe_l2_error, l2_projection};
use formoniq::linalg::faer::{FaerCholesky, FaerLu};
use formoniq::whitney_complex::{HilbertComplex, WhitneyComplex};
use glatt::field::DiffFormClosure;
use regge::mesher::cartesian::CartesianGrid;
use simplicial::atlas::SimplexQuadRule;
use simplicial::{Dim, linalg::Vector};

/// $P_h compose W = id$: the $L^2$ projection is the identity on the Whitney
/// space, since a discrete form is its own best approximation.
///
/// The sharpest available check that the Hodge mass matrix and the source
/// load are the same bilinear form seen from two sides.
#[test]
fn l2_projection_reproduces_whitney_forms() {
  for dim in (1..=3).map(Dim::from) {
    let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
    let lengths = coords.to_edge_lengths_sq(&topology);
    let whitney = WhitneyComplex::new(&topology, &lengths);

    for grade in dim.range_inclusive() {
      let ndofs = topology.nsimplices(grade);
      let cochain = Cochain::new(
        grade,
        Vector::from_iterator(ndofs, (0..ndofs).map(|i| ((i % 7) as f64) - 3.0)),
      );

      let field = WhitneyInterpolant::new(cochain.clone(), &topology);
      let qr = SimplexQuadRule::degree(dim, 3);
      let projected = l2_projection(&field, whitney, Some(qr));

      assert_relative_eq!(projected.coeffs(), cochain.coeffs(), epsilon = 1e-9);
    }
  }
}

/// A form that lies in the Whitney space is reproduced exactly by the
/// projection, and hence has zero $L^2$ error against it: $W$ and $P_h$
/// agree wherever both are exact.
#[test]
fn l2_error_vanishes_on_the_discrete_space() {
  for dim in (1..=3).map(Dim::from) {
    let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
    let lengths = coords.to_edge_lengths_sq(&topology);
    let whitney = WhitneyComplex::new(&topology, &lengths);

    // A globally affine 0-form lies in the Whitney 0-form space.
    let exact = DiffFormClosure::coord_component(0, dim);
    let exact = exact.pullback_on(&topology, &coords);
    let projected = l2_projection(&exact, whitney, Some(SimplexQuadRule::degree(dim, 3)));

    let error = fe_l2_error(&projected, &exact, &topology, &lengths);
    assert!(error < 1e-9, "dim={dim} error={error}");
  }
}

/// The Whitney mass matrix is SPD on a Riemannian geometry, so the mass solve
/// $M c = b$ is a genuine target for conjugate gradients. This pins that the
/// iterative solve agrees with the direct Cholesky factorization to solver
/// tolerance, swept over dimension and grade, the correctness half of
/// wiring `iterative` against a real FEEC operator.
#[test]
fn cg_mass_solve_matches_cholesky() {
  use iterative::{Jacobi, StopCriterion, krylov::cg};

  for dim in (1..=3).map(Dim::from) {
    let (topology, coords) = CartesianGrid::new_unit(dim, 3).triangulate();
    let lengths = coords.to_edge_lengths_sq(&topology);
    let whitney = WhitneyComplex::new(&topology, &lengths);

    for grade in dim.range_inclusive() {
      let mass = whitney.mass(grade);
      let n = mass.nrows();
      let b = Vector::from_fn(n, |i, _| ((i % 5) as f64 - 2.0) * 0.5);

      let direct = FaerCholesky::new(mass.clone()).solve(&b);
      let (iter, report) = cg(&mass, &Jacobi::new(&mass), &b, StopCriterion::rtol(1e-12));

      assert!(report.converged, "dim={dim} grade={grade} did not converge");
      assert!(
        (&iter - &direct).norm() < 1e-9,
        "dim={dim} grade={grade}: cg vs cholesky differ by {}",
        (&iter - &direct).norm()
      );
    }
  }
}

/// Bench, not an assertion: on a well-conditioned mass matrix ($kappa = O(1)$,
/// mesh-independent) Jacobi-CG converges in a fixed handful of iterations, so
/// it competes with the direct factorizations without their fill. Run with
/// `cargo test -p formoniq --release bench_mass_solve -- --nocapture --ignored`.
#[test]
#[ignore = "timing bench, run explicitly with --nocapture"]
fn bench_mass_solve() {
  use iterative::{Jacobi, StopCriterion, krylov::cg};
  use std::time::Instant;

  let dim = Dim::new(3);
  let (topology, coords) = CartesianGrid::new_unit(dim, 12).triangulate();
  let lengths = coords.to_edge_lengths_sq(&topology);
  let whitney = WhitneyComplex::new(&topology, &lengths);

  for grade in dim.range_inclusive() {
    let mass = whitney.mass(grade);
    let n = mass.nrows();
    let b = Vector::from_fn(n, |i, _| ((i % 5) as f64 - 2.0) * 0.5);

    let t = Instant::now();
    let x_lu = FaerLu::new(mass.clone()).solve(&b);
    let t_lu = t.elapsed();

    let t = Instant::now();
    let x_ch = FaerCholesky::new(mass.clone()).solve(&b);
    let t_ch = t.elapsed();

    let precond = Jacobi::new(&mass);
    let t = Instant::now();
    let (x_cg, report) = cg(&mass, &precond, &b, StopCriterion::rtol(1e-10));
    let t_cg = t.elapsed();

    eprintln!(
      "grade {grade}: n={n:>6}  LU {t_lu:>10.2?}  Chol {t_ch:>10.2?}  \
       CG(Jacobi) {t_cg:>10.2?} in {} iters   (agree {:.1e})",
      report.iters,
      (&x_cg - &x_ch).norm().max((&x_lu - &x_ch).norm()),
    );
  }
}
