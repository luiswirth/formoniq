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
//! What the run shows is the *scaling*, not a headline speedup at small $N$: the
//! MG-CG iteration count is flat under refinement while Jacobi-CG's grows with
//! $O(h^(-1))$ and the direct factorization's cost climbs superlinearly. The
//! V-cycle *solve* is cheap and competitive per right-hand side.
//!
//! The honest caveat is the setup. This minimal version builds the Whitney
//! prolongation column by column (see `multigrid::prolongation_matrix`), an
//! $O(N^2)$ cost that dominates the whole run and is what stops the sweep short
//! of the DOF count where MG's $O(N)$ solve overtakes the direct solve. A
//! linear-time prolongation assembled from the subdivision provenance would
//! remove it; until then the crossover is out of reach here, and the takeaway is
//! the iteration-count scaling, not the wall-clock crossover. Run by hand:
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

  // Capped at 4: the O(N^2) prolongation setup, not the solve, is what bounds the
  // sweep. See the module docs.
  for refinements in 1..=4 {
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

    for (name, x) in [("direct", &x_direct), ("jac-cg", &x_jac), ("mg-cg", &x_mg)] {
      let err = (x - &x_true).norm();
      assert!(err < 1e-6, "{name} solution wrong at n = {n}: err {err}");
    }

    println!(
      "{n:>6}  |  {chol_fac:>10.4} {chol_sol:>10.4}  |  \
       {:>4} {jac_setup:>10.4} {jac_sol:>10.4}  |  \
       {:>4} {mg_setup:>10.4} {mg_sol:>10.4}",
      jac_report.iters, mg_report.iters,
    );
  }

  println!("\ntimes in seconds; 'it' is the CG iteration count to rtol {RTOL:e}.");
  println!("the mg-cg iteration count stays flat while jac-cg climbs with 1/h and");
  println!("the direct factorization grows superlinearly. the mg setup is O(N^2)");
  println!("here (column-by-column prolongation) and is the current bottleneck.");
}
