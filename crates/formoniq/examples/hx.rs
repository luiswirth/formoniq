//! Hiptmair-Xu auxiliary-space preconditioning of the grade-k Hodge-Laplace
//! problem, measured against its baselines.
//!
//! For a refinement tower of the unit cube, at a fixed grade, this reports the
//! preconditioned-CG iteration count of
//!
//! - unpreconditioned CG (the raw O(h^-2) conditioning),
//! - Jacobi-CG (a point smoother, which stalls on the dif near-kernel at grade
//!   >= 1),
//! - plain geometric multigrid CG (a V-cycle with a Jacobi smoother, no
//!   auxiliary space -- the thing that is not enough at grade >= 1),
//! - HX-CG with direct auxiliary blocks (the structure, faer-inverted),
//! - HX-CG with multigrid auxiliary blocks (each block a V-cycle),
//!
//! and the wall time of the two HX variants against a direct solve. The claim
//! under test has two parts: only HX keeps the iteration count flat as the mesh
//! is refined at grade >= 1 (both HX columns), and only the multigrid blocks keep
//! the per-iteration cost from being dominated by the direct auxiliary solves, so
//! the HX-MG wall time is what actually scales. There is no theorem for arbitrary
//! (k, N); this is the empirical measurement.
//!
//! Run with `cargo run --release --example hx`.

use std::time::Instant;

use formoniq::{
  hx::GradeKHodgeHx,
  linalg::DirectInverse,
  multigrid::RefinementTower,
  whitney_complex::{HilbertComplex, WhitneyComplex},
};
use iterative::{ApproxInverse, Identity, Jacobi, StopCriterion, krylov::cg};
use simplicial::{
  geometry::coord::mesh::MeshCoords, linalg::Vector, mesher::cartesian::CartesianGrid,
};

/// V-cycle pre- and post-smoothing sweeps, shared by the plain multigrid baseline
/// and the HX multigrid auxiliary blocks.
const SWEEPS: usize = 2;
/// CG stopping tolerance and iteration cap (so an ill-conditioned baseline
/// reports its failure to converge rather than running unboundedly).
const RTOL: f64 = 1e-8;
const MAX_ITERS: usize = 2000;

/// The intrinsic refinement tower together with the finest-level coordinates the
/// HX vector-nodal space reads its ambient frame off (invariant 2).
struct Bench {
  tower: RefinementTower,
  finest_coords: MeshCoords,
}

impl Bench {
  fn build(dim: usize, base: usize, levels: usize) -> Self {
    let (base_topology, base_coords) = CartesianGrid::new_unit(dim, base).triangulate();
    let base_geometry = base_coords.to_edge_lengths_sq(&base_topology);
    let tower = RefinementTower::new(base_topology, base_geometry, levels);
    let finest_coords = tower
      .subdivisions()
      .iter()
      .fold(base_coords, |c, sub| c.refine(sub));
    Self {
      tower,
      finest_coords,
    }
  }

  fn finest(&self) -> WhitneyComplex<'_> {
    self.tower.finest_whitney()
  }
}

fn stop() -> StopCriterion {
  StopCriterion {
    rtol: RTOL,
    max_iters: MAX_ITERS,
  }
}

/// Format an iteration count, marking non-convergence within the cap.
fn iters(report: &iterative::Report) -> String {
  if report.converged {
    report.iters.to_string()
  } else {
    format!(">{MAX_ITERS}")
  }
}

/// Time a single application of `f`, returning its result and the wall time in
/// milliseconds.
fn timed<T>(f: impl FnOnce() -> T) -> (T, f64) {
  let start = Instant::now();
  let value = f();
  (value, start.elapsed().as_secs_f64() * 1e3)
}

fn benchmark(dim: usize, grade: usize, base: usize, max_levels: usize) {
  println!("\n=== dim {dim}, grade {grade} (unit {dim}-cube) ===");
  println!(
    "{:>7}  {:>7}  {:>7}  {:>7}  {:>7}  {:>7}   {:>9}  {:>9}  {:>9}",
    "ndofs", "none", "jacobi", "mg", "hx", "hx-mg", "hx [ms]", "hxmg [ms]", "direct [ms]"
  );

  for levels in 1..=max_levels {
    let bench = Bench::build(dim, base, levels);
    let complex = bench.finest();
    let operator = complex.hdif_gram(grade);
    let n = operator.nrows();
    let rhs = Vector::from_fn(n, |i, _| ((i * i + 1) as f64).cos());

    let none = cg(&operator, &Identity::new(n), &rhs, stop()).1;
    let jacobi = cg(&operator, &Jacobi::new(&operator), &rhs, stop()).1;
    let mg = cg(
      &operator,
      &bench.tower.grade_vcycle(grade, SWEEPS),
      &rhs,
      stop(),
    )
    .1;

    let hx = GradeKHodgeHx::new(&complex, &bench.finest_coords, grade);
    let (hx_report, hx_ms) = timed(|| hx.solve(&rhs, stop()).1);

    let hxmg = GradeKHodgeHx::with_multigrid(&bench.tower, &bench.finest_coords, grade, SWEEPS);
    let (hxmg_report, hxmg_ms) = timed(|| hxmg.solve(&rhs, stop()).1);

    let (_, direct_ms) = timed(|| {
      DirectInverse::try_new(operator.clone())
        .unwrap()
        .apply(&rhs)
    });

    println!(
      "{n:>7}  {:>7}  {:>7}  {:>7}  {:>7}  {:>7}   {hx_ms:>9.1}  {hxmg_ms:>9.1}  {direct_ms:>9.1}",
      iters(&none),
      iters(&jacobi),
      iters(&mg),
      iters(&hx_report),
      iters(&hxmg_report),
    );
  }
}

fn main() {
  // Grade 1 in 2D and 3D (the classic H(curl) case), and grade 2 in 3D (H(div)),
  // the two grades where the dif near-kernel is what stalls a point smoother.
  benchmark(2, 1, 2, 6);
  benchmark(3, 1, 1, 4);
  benchmark(3, 2, 1, 4);
}
