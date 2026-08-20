//! [`Grade0Multigrid`]: MG-CG matches the direct solve, the Galerkin coarse
//! operator equals reassembly on the coarse mesh, and the iteration count
//! stays mesh-independent.

use derham::prolongate::prolongation_matrix;
use formoniq::linalg::DirectInverse;
use formoniq::multigrid::Grade0Multigrid;
use formoniq::whitney_complex::{HilbertComplex, WhitneyComplex};
use iterative::{ApproxInverse, Identity, StopCriterion, krylov::cg};
use regge::lengths::mesh::MeshLengthsSq;
use regge::mesher::cartesian::CartesianGrid;
use simplicial::linalg::Vector;
use simplicial::topology::complex::Complex;
use simplicial::topology::ordering::CellOrdering;

/// A 2D unit-square tower: a base grid of `base` cells per axis, refined
/// `refinements` times. Returns the coarse topology and geometry the builder
/// consumes. Colex refinement composes in 2D (invariant 7).
fn unit_square(base: usize) -> (Complex, MeshLengthsSq) {
  let (topology, coords) = CartesianGrid::new_unit(2, base).triangulate();
  let geometry = coords.to_edge_lengths_sq(&topology);
  (topology, geometry)
}

/// MG-CG reproduces the direct solve of the same finest-level system: the
/// preconditioner changes the path, never the fixed point.
#[test]
fn mg_cg_matches_the_direct_solve() {
  let (topology, geometry) = unit_square(2);
  let mg = Grade0Multigrid::new(topology, geometry, 3, 2);

  let n = mg.fine_operator().nrows();
  let rhs = Vector::from_fn(n, |i, _| ((i * i) as f64).cos());

  let (x_mg, report) = mg.solve(&rhs, StopCriterion::rtol(1e-10));
  assert!(report.converged, "MG-CG did not converge");

  let direct = DirectInverse::try_new(mg.fine_operator().clone()).unwrap();
  let x_direct = direct.apply(&rhs);
  assert!(
    (&x_mg - &x_direct).norm() < 1e-8,
    "MG-CG disagrees with direct: {}",
    (&x_mg - &x_direct).norm()
  );
}

/// The Galerkin coarse operator $P^T A_f P$ equals the operator reassembled on
/// the coarse mesh, at grade 0. This is what makes the coarse correction a
/// consistent discretization and not merely an algebraic reduction. The
/// Whitney prolongation is exact and metric-free, so the two agree to rounding.
#[test]
fn galerkin_coarse_matches_reassembly() {
  let (topology, geometry) = unit_square(2);
  let coarse = WhitneyComplex::new(&topology, &geometry);
  let a_coarse = coarse.hdif_gram(0);

  let ordering = CellOrdering::colex(&topology);
  let sub = topology.refine_with(&ordering, 2);
  let fine_geometry = geometry.refine(&sub, &topology);
  let p = prolongation_matrix(0, &topology, &sub);

  let a_fine = WhitneyComplex::new(sub.complex(), &fine_geometry).hdif_gram(0);
  let galerkin = &p.transpose() * &(&a_fine * &p);

  let diff = &galerkin - &a_coarse;
  let frob: f64 = diff
    .triplet_iter()
    .map(|(_, _, v)| v * v)
    .sum::<f64>()
    .sqrt();
  let scale: f64 = a_coarse
    .triplet_iter()
    .map(|(_, _, v)| v * v)
    .sum::<f64>()
    .sqrt();
  assert!(
    frob < 1e-10 * scale,
    "Galerkin != reassembly: {frob} vs {scale}"
  );
}

/// The MG-CG iteration count stays essentially flat as the mesh is refined,
/// while unpreconditioned CG grows with the $O(h^(-2))$ condition number,
/// the mesh-independence multigrid exists to provide.
#[test]
fn mg_cg_iterations_are_mesh_independent() {
  let iters = |refinements: usize| -> (usize, usize) {
    let (topology, geometry) = unit_square(2);
    let mg = Grade0Multigrid::new(topology, geometry, refinements, 2);
    let n = mg.fine_operator().nrows();
    let rhs = Vector::from_fn(n, |i, _| (i as f64 + 1.0).ln());
    let stop = StopCriterion::rtol(1e-10);
    let (_, mg_report) = mg.solve(&rhs, stop);
    let (_, plain_report) = cg(mg.fine_operator(), &Identity::new(n), &rhs, stop);
    (mg_report.iters, plain_report.iters)
  };
  let (mg_coarse, _) = iters(2);
  let (mg_fine, plain_fine) = iters(4);
  assert!(
    mg_fine <= mg_coarse + 3,
    "MG-CG count grew under refinement: {mg_coarse} -> {mg_fine}"
  );
  assert!(
    mg_fine * 3 < plain_fine,
    "MG-CG ({mg_fine}) not decisively beating plain CG ({plain_fine})"
  );
}
