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
//! - HX-CG (the auxiliary-space preconditioner),
//!
//! and the wall time of HX-CG against a direct solve, to see the crossover. The
//! claim under test is that only HX keeps the iteration count flat as the mesh is
//! refined at grade >= 1. There is no theorem for arbitrary (k, N); this is the
//! empirical measurement.
//!
//! Run with `cargo run --release --example hx`.

use std::time::Instant;

use derham::prolongate::prolongation_matrix;
use formoniq::{
  hx::GradeKHodgeHx,
  linalg::DirectInverse,
  whitney_complex::{HilbertComplex, WhitneyComplex},
};
use iterative::{ApproxInverse, Identity, Jacobi, Level, StopCriterion, VCycle, krylov::cg};
use simplicial::{
  geometry::{coord::mesh::MeshCoords, metric::mesh::MeshLengthsSq},
  linalg::{CsrMatrix, Vector},
  mesher::cartesian::CartesianGrid,
  topology::{complex::Complex, ordering::CellOrdering, refine::Subdivision},
};

/// Damped-Jacobi smoother weight for the multigrid and HX smoothers.
const SMOOTHER_WEIGHT: f64 = 2.0 / 3.0;
/// V-cycle pre- and post-smoothing sweeps.
const SWEEPS: usize = 2;
/// CG stopping tolerance and iteration cap (so an ill-conditioned baseline
/// reports its failure to converge rather than running unboundedly).
const RTOL: f64 = 1e-8;
const MAX_ITERS: usize = 2000;

/// A refinement tower of the unit `dim`-cube: `levels` colex-refinements of a
/// base grid, each level its complex, coordinates and the subdivision that bore
/// it from the coarser one.
struct Tower {
  complexes: Vec<Complex>,
  geometries: Vec<MeshLengthsSq>,
  coords: Vec<MeshCoords>,
  subdivisions: Vec<Subdivision>,
}

impl Tower {
  fn build(dim: usize, base: usize, levels: usize) -> Self {
    let (base_topology, base_coords) = CartesianGrid::new_unit(dim, base).triangulate();
    let mut complexes = vec![base_topology];
    let mut coords = vec![base_coords];
    let mut subdivisions = Vec::new();
    let mut ordering = CellOrdering::colex(&complexes[0]);

    for _ in 0..levels {
      let sub = complexes.last().unwrap().refine_with(&ordering, 2);
      let fine_coords = coords.last().unwrap().refine(&sub);
      ordering = sub.ordering().clone();
      complexes.push(sub.complex().clone());
      coords.push(fine_coords);
      subdivisions.push(sub);
    }

    let geometries = complexes
      .iter()
      .zip(&coords)
      .map(|(topology, coords)| coords.to_edge_lengths_sq(topology))
      .collect();

    Self {
      complexes,
      geometries,
      coords,
      subdivisions,
    }
  }

  fn finest(&self) -> usize {
    self.complexes.len() - 1
  }

  fn whitney(&self, level: usize) -> WhitneyComplex<'_> {
    WhitneyComplex::new(&self.complexes[level], &self.geometries[level])
  }

  /// A plain geometric multigrid V-cycle for grade `grade` over the whole tower:
  /// Galerkin-free reassembled operators, Whitney prolongation transfers, a
  /// damped-Jacobi smoother, and a direct coarse solve. No auxiliary space.
  fn grade_k_vcycle(&self, grade: usize) -> VCycle<Jacobi, DirectInverse> {
    let operators: Vec<CsrMatrix> = (0..self.complexes.len())
      .map(|l| self.whitney(l).hdif_gram(grade))
      .collect();

    let levels: Vec<Level<Jacobi>> = (1..operators.len())
      .rev()
      .map(|f| {
        let prolong = prolongation_matrix(grade, &self.complexes[f - 1], &self.subdivisions[f - 1]);
        let restrict = prolong.transpose();
        let smoother = Jacobi::weighted(&operators[f], SMOOTHER_WEIGHT);
        Level::new(operators[f].clone(), smoother, prolong, restrict)
      })
      .collect();

    let coarse =
      DirectInverse::try_new(operators[0].clone()).expect("coarsest operator must be SPD");
    VCycle::symmetric(levels, coarse, SWEEPS)
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
    format!(">{}", MAX_ITERS)
  }
}

fn benchmark(dim: usize, grade: usize, base: usize, max_levels: usize) {
  println!("\n=== dim {dim}, grade {grade} (unit {dim}-cube) ===");
  println!(
    "{:>7}  {:>8}  {:>8}  {:>8}  {:>8}   {:>10}  {:>10}",
    "ndofs", "none", "jacobi", "mg", "hx", "hx [ms]", "direct [ms]"
  );

  for levels in 1..=max_levels {
    let tower = Tower::build(dim, base, levels);
    let finest = tower.finest();
    let complex = tower.whitney(finest);
    let operator = complex.hdif_gram(grade);
    let n = operator.nrows();
    let rhs = Vector::from_fn(n, |i, _| ((i * i + 1) as f64).cos());

    let none = cg(&operator, &Identity::new(n), &rhs, stop()).1;
    let jacobi = cg(&operator, &Jacobi::new(&operator), &rhs, stop()).1;
    let mg = cg(&operator, &tower.grade_k_vcycle(grade), &rhs, stop()).1;

    let hx = GradeKHodgeHx::new(&complex, &tower.coords[finest], grade);
    let start = Instant::now();
    let (_, hx_report) = hx.solve(&rhs, stop());
    let hx_ms = start.elapsed().as_secs_f64() * 1e3;

    let start = Instant::now();
    let _ = DirectInverse::try_new(operator.clone())
      .unwrap()
      .apply(&rhs);
    let direct_ms = start.elapsed().as_secs_f64() * 1e3;

    println!(
      "{n:>7}  {:>8}  {:>8}  {:>8}  {:>8}   {hx_ms:>10.1}  {direct_ms:>10.1}",
      iters(&none),
      iters(&jacobi),
      iters(&mg),
      iters(&hx_report),
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
