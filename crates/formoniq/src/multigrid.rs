//! Geometric multigrid for the grade-0 Hodge-Laplace problem.
//!
//! The FEEC wiring of the generic [`iterative::VCycle`]: a refinement tower of
//! Whitney complexes supplies the levels, the operator on each is the grade-0
//! Hilbert-space Gram matrix $M_0 + D_0^T M_1 D_0$ (the mass plus the up-stiffness
//! of the scalar Hodge-Laplacian, SPD on a Riemannian geometry), and the
//! intergrid transfer is the Whitney prolongation $P$ of [`derham::prolongate`]
//! with restriction $R = P^T$. The coarse solver is the direct faer
//! factorization ([`crate::linalg::DirectInverse`]).
//!
//! This is the minimal, nodal case: a pointwise (Jacobi) smoother already damps
//! the high-frequency error, since grade 0 has no large near-kernel of $dif$ to
//! confound it. Higher grades need the auxiliary-space / regular-decomposition
//! smoother the grade-mixed near-kernel calls for; that is deliberately out of
//! scope here, and the cycle machinery it will reuse is exactly this one.
//!
//! The coarse operators are formed by the Galerkin triple product
//! $A_c = P^T A_f P$ rather than reassembled on the coarse mesh: it is defined by
//! the transfer and the fine operator alone, which is what keeps the cycle
//! symmetric and the two-grid analysis clean. That the two agree at grade 0 is a
//! test, not an assumption.

use derham::prolongate::prolongation_matrix;
use iterative::{
  Jacobi, Level, VCycle,
  krylov::cg,
  {Report, StopCriterion},
};
use simplicial::{
  geometry::metric::mesh::MeshLengthsSq,
  linalg::{CsrMatrix, Vector},
  topology::{complex::Complex, ordering::CellOrdering},
};

use crate::{
  linalg::DirectInverse,
  whitney_complex::{HilbertComplex, WhitneyComplex},
};

/// The damping factor of the Jacobi smoother. Two-thirds is the classic optimum
/// for a second-order operator on a regular grid: it damps the upper half of the
/// spectrum, which is the error a coarser level cannot represent.
const SMOOTHER_WEIGHT: f64 = 2.0 / 3.0;

/// A grade-0 multigrid solver built on a refinement tower.
///
/// Owns the tower (complexes and geometries, coarse to fine) so a right-hand
/// side can be assembled on the finest level, the finest operator, and the
/// V-cycle preconditioner. [`solve`](Self::solve) runs V-cycle-preconditioned CG
/// on the finest level.
pub struct Grade0Multigrid {
  complexes: Vec<Complex>,
  geometries: Vec<MeshLengthsSq>,
  fine_operator: CsrMatrix,
  cycle: VCycle<Jacobi, DirectInverse>,
}

impl Grade0Multigrid {
  /// Build the tower by refining `base_topology`/`base_geometry` `refinements`
  /// times (halving the mesh each step), and assemble the V-cycle over it with
  /// `sweeps` pre- and post-smoothing steps.
  ///
  /// The base ordering is colex; each refined level inherits the ordering the
  /// `Subdivision` carries, so the tower composes (invariant 7). Refinement is
  /// metric-free and exact --- a flat cell subdivided stays flat --- so the
  /// tower introduces no geometric error of its own.
  ///
  /// # Panics
  /// If a level's operator is not positive definite (a non-Riemannian geometry),
  /// which the direct coarse solve requires.
  pub fn new(
    base_topology: Complex,
    base_geometry: MeshLengthsSq,
    refinements: usize,
    sweeps: usize,
  ) -> Self {
    let mut complexes = vec![base_topology];
    let mut geometries = vec![base_geometry];
    // prolongations[d] transfers level d (coarse) into level d + 1 (fine).
    let mut prolongations: Vec<CsrMatrix> = Vec::new();
    let mut ordering = CellOrdering::colex(&complexes[0]);

    for _ in 0..refinements {
      let coarse = complexes.last().unwrap();
      let sub = coarse.refine_with(&ordering, 2);
      let fine_geometry = geometries.last().unwrap().refine(&sub, coarse);
      let prolongation = prolongation_matrix(0, coarse, &sub);
      ordering = sub.ordering().clone();
      let fine = sub.into_complex();
      prolongations.push(prolongation);
      complexes.push(fine);
      geometries.push(fine_geometry);
    }

    let operators: Vec<CsrMatrix> = complexes
      .iter()
      .zip(&geometries)
      .map(|(topology, geometry)| WhitneyComplex::new(topology, geometry).hdif_gram(0))
      .collect();

    // Levels finest first: for each fine level f, the transfer is prolongations
    // [f - 1] and its transpose.
    let n = operators.len();
    let levels: Vec<Level<Jacobi>> = (1..n)
      .rev()
      .map(|f| {
        let prolong = prolongations[f - 1].clone();
        let restrict = prolong.transpose();
        let smoother = Jacobi::weighted(&operators[f], SMOOTHER_WEIGHT);
        Level::new(operators[f].clone(), smoother, prolong, restrict)
      })
      .collect();

    let coarse =
      DirectInverse::try_new(operators[0].clone()).expect("coarsest grade-0 operator must be SPD");
    let cycle = VCycle::symmetric(levels, coarse, sweeps);

    Self {
      fine_operator: operators.into_iter().next_back().unwrap(),
      complexes,
      geometries,
      cycle,
    }
  }

  /// The finest Whitney complex, for assembling a right-hand side.
  pub fn fine_complex(&self) -> WhitneyComplex<'_> {
    let last = self.complexes.len() - 1;
    WhitneyComplex::new(&self.complexes[last], &self.geometries[last])
  }

  /// The finest-level operator $M_0 + D_0^T M_1 D_0$.
  pub fn fine_operator(&self) -> &CsrMatrix {
    &self.fine_operator
  }

  /// The V-cycle preconditioner, exposed so it can be compared against other
  /// preconditioners or iterated on its own.
  pub fn cycle(&self) -> &VCycle<Jacobi, DirectInverse> {
    &self.cycle
  }

  /// Solve `fine_operator x = rhs` by V-cycle-preconditioned CG on the finest
  /// level.
  pub fn solve(&self, rhs: &Vector, stop: StopCriterion) -> (Vector, Report) {
    cg(&self.fine_operator, &self.cycle, rhs, stop)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use iterative::{ApproxInverse, Identity, krylov::cg};
  use simplicial::mesher::cartesian::CartesianGrid;

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
  /// while unpreconditioned CG grows with the $O(h^(-2))$ condition number ---
  /// the mesh-independence that is the whole point of multigrid.
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
}
