//! Multigrid against the naive solvers, on the grade-0 Hodge-Laplace system
//! $M_0 + D_0^T M_1 D_0$ over a refinement tower of the 2D unit square.
//!
//! Three solvers on the same finest-level system, timed as the mesh refines:
//!
//! - the direct sparse Cholesky (`formoniq::linalg::DirectInverse`), the current
//!   default and the reference to beat;
//! - Jacobi-preconditioned CG, the naive iterative baseline, whose iteration
//!   count grows with the $O(h^(-2))$ condition number;
//! - V-cycle-preconditioned CG, whose iteration count is mesh-independent.
//!
//! Two things the numbers make concrete. MG-CG decisively beats the naive
//! iterative solver: at the finest level its solve is more than an order of
//! magnitude faster than Jacobi-CG, whose iteration count has grown into the
//! thousands. Against the *direct* solve it has not yet won in wall clock: in 2D
//! the sparse Cholesky is very strong (near-linear fill under nested dissection,
//! a tiny back-solve), so the crossover sits beyond these sizes -- it is in 3D,
//! where the direct factorization is $O(N^2)$, or under many right-hand sides,
//! that MG's $O(N)$ solve pays off. What the run does show is the scaling that
//! guarantees the crossover exists: the MG-CG iteration count is flat.
//!
//! The setup is now linear in the DOF count (the Whitney prolongation is
//! assembled in one pass, #118); what remains of it is the finest-level operator
//! assembly, which the direct solve reuses for free, so the setup column
//! overstates MG's true marginal cost. Run by hand:
//!
//! ```sh
//! cargo run --release --example multigrid
//! ```

use std::time::Instant;

use formoniq::{linalg::DirectInverse, multigrid::Grade0Multigrid};
use iterative::{ApproxInverse, Jacobi, StopCriterion, krylov::cg};
use simplicial::{
  geometry::metric::mesh::MeshLengthsSq,
  linalg::{CsrMatrix, Vector},
  mesher::cartesian::CartesianGrid,
  topology::complex::Complex,
};

const BASE_CELLS_PER_AXIS: usize = 4;
const SMOOTHING_SWEEPS: usize = 2;
const RTOL: f64 = 1e-10;

fn unit_square() -> (Complex, MeshLengthsSq) {
  let (topology, coords) = CartesianGrid::new_unit(2, BASE_CELLS_PER_AXIS).triangulate();
  let geometry = coords.to_edge_lengths_sq(&topology);
  (topology, geometry)
}

/// Wall-clock a closure, returning its result and the elapsed seconds.
fn timed<T>(f: impl FnOnce() -> T) -> (T, f64) {
  let start = Instant::now();
  let value = f();
  (value, start.elapsed().as_secs_f64())
}

fn main() {
  tracing_subscriber::fmt::init();

  println!(
    "{:>6}  |  {:>10} {:>10}  |  {:>4} {:>10} {:>10}  |  {:>4} {:>10} {:>10}",
    "dofs", "chol fac", "chol sol", "it", "jac setup", "jac sol", "it", "mg setup", "mg sol"
  );
  println!("{}", "-".repeat(96));

  let stop = StopCriterion::rtol(RTOL);

  for refinements in 1..=7 {
    let (topology, geometry) = unit_square();

    // The multigrid solver owns the tower; its finest operator is the shared
    // system every method solves, so the comparison is apples to apples.
    let (mg, mg_setup) =
      timed(|| Grade0Multigrid::new(topology, geometry, refinements, SMOOTHING_SWEEPS));
    let a: &CsrMatrix = mg.fine_operator();
    let n = a.nrows();

    // A manufactured right-hand side with a known solution, so every method is
    // checked to reproduce it, not merely to converge.
    let x_true = Vector::from_fn(n, |i, _| ((i as f64 + 1.0) * 0.5).sin());
    let b = a * &x_true;

    let (direct, chol_fac) = timed(|| DirectInverse::try_new(a.clone()).expect("SPD"));
    let (x_direct, chol_sol) = timed(|| direct.apply(&b));

    let (jacobi, jac_setup) = timed(|| Jacobi::new(a));
    let ((x_jac, jac_report), jac_sol) = timed(|| cg(a, &jacobi, &b, stop));

    let ((x_mg, mg_report), mg_sol) = timed(|| mg.solve(&b, stop));

    // A residual tolerance does not tightly bound the solution error on an
    // ill-conditioned system, so the iterative solvers are held to a looser
    // relative error than the direct solve; all three must still reproduce the
    // manufactured solution.
    let rel = |x: &Vector| (x - &x_true).norm() / x_true.norm();
    assert!(rel(&x_direct) < 1e-9, "direct solution wrong at n = {n}");
    for (name, x) in [("jac-cg", &x_jac), ("mg-cg", &x_mg)] {
      assert!(
        rel(x) < 1e-4,
        "{name} solution wrong at n = {n}: rel {}",
        rel(x)
      );
    }

    println!(
      "{n:>6}  |  {chol_fac:>10.4} {chol_sol:>10.4}  |  \
       {:>4} {jac_setup:>10.4} {jac_sol:>10.4}  |  \
       {:>4} {mg_setup:>10.4} {mg_sol:>10.4}",
      jac_report.iters, mg_report.iters,
    );
  }

  println!("\ntimes in seconds; 'it' is the CG iteration count to rtol {RTOL:e}.");
  println!("mg-cg iterations stay flat while jac-cg climbs with 1/h; mg-cg's solve");
  println!("beats jac-cg by an order of magnitude. the 2D direct solve is still");
  println!("faster in wall clock here -- the crossover is in 3D or under many rhs.");
}
