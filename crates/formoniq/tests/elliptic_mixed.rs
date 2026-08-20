//! The mixed Hodge-Laplace saddle point: block-preconditioned MINRES
//! matches the direct LU factorization, and its iteration count stays
//! mesh-independent (Arnold-Falk-Winther norm equivalence).

use formoniq::galerkin::GalerkinVector;
use formoniq::linalg::faer::FaerLu;
use formoniq::problems::elliptic::{
  assemble_mixed_kkt, mixed_block_preconditioner, solve_harmonics,
};
use formoniq::whitney_complex::WhitneyComplex;
use iterative::StopCriterion;
use iterative::krylov::minres;
use regge::mesher::cartesian::CartesianGrid;
use simplicial::{Dim, linalg::Vector};

/// Block-preconditioned MINRES solves the mixed Hodge-Laplace KKT system to
/// the same solution as the direct LU factorization, swept over dimension and
/// grade on a Riemannian mesh where the diagonal blocks are SPD.
#[test]
fn block_minres_matches_direct_on_the_mixed_system() {
  for dim in (1..=3).map(Dim::from) {
    let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
    let lengths = coords.to_edge_lengths_sq(&topology);
    let relative = WhitneyComplex::new(&topology, &lengths).relative();

    for grade in dim.range_inclusive() {
      let harmonics = solve_harmonics(&relative, grade).unwrap();
      let source = GalerkinVector::new(
        grade,
        Vector::from_fn(topology.nsimplices(grade), |i, _| (i % 7) as f64 - 3.0),
      );
      let (a, b, _, _) = assemble_mixed_kkt(&relative, source, grade, &harmonics);

      let precond = mixed_block_preconditioner(&relative, grade, harmonics.ncols())
        .expect("SPD blocks on a Riemannian geometry");
      let (x_min, report) = minres(&a, &precond, &b, StopCriterion::rtol(1e-11));
      assert!(report.converged, "dim={dim} grade={grade} did not converge");

      let x_lu = FaerLu::new(a.clone()).solve(&b);
      let err = (&x_min - &x_lu).norm() / x_lu.norm().max(1.0);
      assert!(
        err < 1e-7,
        "dim={dim} grade={grade}: block-MINRES vs LU {err:.1e}"
      );
    }
  }
}

/// Bench, not an assertion: block-preconditioned MINRES against the direct LU
/// on the mixed Hodge-Laplace system. Both are timed end to end, the
/// iterative side pays for factoring its SPD blocks, the direct side for
/// factoring the whole indefinite system. Run with
/// `cargo test -p formoniq --release bench_mixed_solve -- --nocapture --ignored`.
#[test]
#[ignore = "timing bench, run explicitly with --nocapture"]
fn bench_mixed_solve() {
  use std::time::Instant;

  for (dim, refinement) in [(Dim::new(2), 24usize), (Dim::new(3), 8)] {
    let (topology, coords) = CartesianGrid::new_unit(dim, refinement).triangulate();
    let lengths = coords.to_edge_lengths_sq(&topology);
    let relative = WhitneyComplex::new(&topology, &lengths).relative();

    for grade in dim.range_inclusive() {
      let harmonics = solve_harmonics(&relative, grade).unwrap();
      let source = GalerkinVector::new(
        grade,
        Vector::from_fn(topology.nsimplices(grade), |i, _| (i % 7) as f64 - 3.0),
      );
      let (a, b, _, _) = assemble_mixed_kkt(&relative, source, grade, &harmonics);
      let n = a.nrows();

      let t = Instant::now();
      let x_lu = FaerLu::new(a.clone()).solve(&b);
      let t_lu = t.elapsed();

      let t = Instant::now();
      let precond = mixed_block_preconditioner(&relative, grade, harmonics.ncols()).unwrap();
      let t_setup = t.elapsed();
      let t = Instant::now();
      let (x_min, report) = minres(&a, &precond, &b, StopCriterion::rtol(1e-10));
      let t_iter = t.elapsed();

      eprintln!(
        "dim {dim} grade {grade}: n={n:>7}  LU {t_lu:>10.2?}  \
         block-MINRES {:>10.2?} (setup {t_setup:>9.2?} + {t_iter:>9.2?}, {} iters)  agree {:.1e}",
        t_setup + t_iter,
        report.iters,
        (&x_min - &x_lu).norm() / x_lu.norm().max(1.0),
      );
    }
  }
}

/// The point of the block preconditioner: the MINRES iteration count stays
/// bounded under mesh refinement (Arnold-Falk-Winther norm equivalence),
/// rather than growing like the unpreconditioned condition number.
#[test]
fn block_minres_iteration_count_is_mesh_independent() {
  let grade = Dim::new(1);
  let iters_at = |refinement: usize| {
    let (topology, coords) = CartesianGrid::new_unit(Dim::new(2), refinement).triangulate();
    let lengths = coords.to_edge_lengths_sq(&topology);
    let relative = WhitneyComplex::new(&topology, &lengths).relative();
    let harmonics = solve_harmonics(&relative, grade).unwrap();
    let source = GalerkinVector::new(
      grade,
      Vector::from_fn(topology.nsimplices(grade), |i, _| {
        ((i % 5) as f64 - 2.0) * 0.3
      }),
    );
    let (a, b, _, _) = assemble_mixed_kkt(&relative, source, grade, &harmonics);
    let precond = mixed_block_preconditioner(&relative, grade, harmonics.ncols()).unwrap();
    let (_, report) = minres(&a, &precond, &b, StopCriterion::rtol(1e-10));
    assert!(report.converged);
    report.iters
  };

  let coarse = iters_at(4);
  let fine = iters_at(8); // 4x the DOFs
  // Bounded, not growing with h: the unpreconditioned count would roughly
  // double. A small additive slack absorbs discretization variation.
  assert!(
    fine <= coarse + 8,
    "coarse={coarse} fine={fine}: not mesh-independent"
  );
}
