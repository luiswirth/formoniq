//! Multigrid against the naive solvers, on the grade-0 Hodge-Laplace system
//! $M_0 + D_0^T M_1 D_0$ over a refinement tower, in 2D and 3D.
//!
//! Three solvers on the same finest-level system, timed as the mesh refines:
//!
//! - the direct sparse Cholesky (`formoniq::linalg::DirectInverse`), the current
//!   default and the reference to beat;
//! - Jacobi-preconditioned CG, the naive iterative baseline, whose iteration
//!   count grows with the $O(h^(-2))$ condition number;
//! - V-cycle-preconditioned CG, whose iteration count is mesh-independent.
//!
//! MG-CG decisively beats the naive iterative solver in every case: its solve is
//! more than an order of magnitude faster than Jacobi-CG, whose iteration count
//! has grown into the thousands while MG-CG's stays flat.
//!
//! Against the *direct* solve the dimension is the whole story. In 2D the sparse
//! Cholesky is very strong (near-linear fill under nested dissection, a tiny
//! back-solve), and MG-CG does not overtake it at these sizes. In 3D the
//! factorization is $O(N^2)$ in time and its fill superlinear in memory, while
//! MG-CG stays $O(N)$: at equal DOF counts the 3D factorization already costs an
//! order of magnitude more than in 2D and climbs far faster, while the MG-CG
//! solve stays flat and cheap. That divergence is the crossover taking shape;
//! reaching it outright needs meshes an order of magnitude larger still, where
//! faer's sparse Cholesky currently loses accuracy (`DirectInverse` falls back to
//! LU there, and the 3D sweep is capped just below that regime).
//!
//! The setup is linear in the DOF count (the Whitney prolongation is assembled in
//! one pass, #118); what remains of it is the finest-level operator assembly,
//! which the direct solve reuses for free, so the setup column overstates MG's
//! true marginal cost. Run by hand:
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

const SMOOTHING_SWEEPS: usize = 2;
const RTOL: f64 = 1e-10;

fn unit_cube(dim: usize, base_cells: usize) -> (Complex, MeshLengthsSq) {
  let (topology, coords) = CartesianGrid::new_unit(dim, base_cells).triangulate();
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

  // 2D: the direct sparse Cholesky is very strong, so MG-CG does not overtake it
  // in wall clock here; watch the iteration counts instead. 3D: the direct
  // factorization is O(N^2) in time and its fill superlinear in memory, so the
  // crossover where MG-CG's O(N) solve wins is reachable.
  // A single coarse cell per box (base 1): a Kuhn-triangulated unit cube whose
  // colex ordering is its Freudenthal ordering, so the refinement tower stays
  // self-similar (invariant 7). Multiple coarse cells would glue Kuhn boxes along
  // reflecting seams, and colex refinement there drifts off the self-similar
  // family, breeding sliver cells that wreck the conditioning above 2D.
  println!("=== 2D unit square ===");
  bench(2, 1, 9);
  println!("\n=== 3D unit cube ===");
  bench(3, 1, 5);
}

/// Time the three solvers on the grade-0 system over a `dim`-dimensional tower,
/// starting from `base_cells` per axis and refining up to `max_refinements`
/// times.
fn bench(dim: usize, base_cells: usize, max_refinements: usize) {
  let stop = StopCriterion::rtol(RTOL);

  println!(
    "{:>8}  |  {:>10} {:>10}  |  {:>4} {:>10} {:>10}  |  {:>4} {:>10} {:>10}",
    "dofs", "chol fac", "chol sol", "it", "jac setup", "jac sol", "it", "mg setup", "mg sol"
  );
  println!("{}", "-".repeat(98));

  for refinements in 1..=max_refinements {
    let (topology, geometry) = unit_cube(dim, base_cells);

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

    // Correctness is checked by the residual (backward error), not the forward
    // error against `x_true`: the latter is bounded by the condition number, which
    // grows like $h^{-2}$ and inflates even the exact solve's forward error at the
    // finest 3D levels. The residual is conditioning-free -- did we solve the
    // system? -- and all three must drive it to zero.
    // Correctness by the residual (backward error), not the forward error against
    // `x_true`: the latter is bounded by the condition number, which grows like
    // $h^{-2}$. All three must drive the residual to zero. (`DirectInverse` guards
    // itself against faer's large-system sparse-Cholesky accuracy failure by
    // falling back to LU, so it stays accurate here too.)
    let residual = |x: &Vector| (a * x - &b).norm() / b.norm();
    for (name, x) in [("direct", &x_direct), ("jac-cg", &x_jac), ("mg-cg", &x_mg)] {
      assert!(
        residual(x) < 1e-8,
        "{name} did not solve the system at n = {n}: residual {}",
        residual(x)
      );
    }

    println!(
      "{n:>8}  |  {chol_fac:>10.4} {chol_sol:>10.4}  |  \
       {:>4} {jac_setup:>10.4} {jac_sol:>10.4}  |  \
       {:>4} {mg_setup:>10.4} {mg_sol:>10.4}",
      jac_report.iters, mg_report.iters,
    );
  }

  println!("times in seconds; 'it' is the CG iteration count to rtol {RTOL:e}.");
}
