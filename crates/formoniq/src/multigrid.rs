//! Geometric multigrid for the grade-$k$ Hodge-Laplace problem.
//!
//! The FEEC wiring of the generic [`iterative::VCycle`]: a [`RefinementTower`] of
//! Whitney complexes supplies the levels, the operator on each is the
//! Hilbert-space Gram matrix $A_k = M_k + D_k^T M_(k+1) D_k$ (the mass plus the
//! up-stiffness of the Hodge-Laplacian, [`HilbertComplex::hdif_gram`], SPD on a
//! Riemannian geometry), and the intergrid transfer is the Whitney prolongation
//! $P$ of [`derham::prolongate`] with restriction $R = P^T$. The coarse solver is
//! the direct faer factorization ([`crate::linalg::DirectInverse`]).
//!
//! At grade $0$ this is the minimal, nodal case: a pointwise (Jacobi) smoother
//! already damps the high-frequency error, since grade 0 has no large near-kernel
//! of $dif$ to confound it, and [`Grade0Multigrid`] is exactly that. At grade
//! $>= 1$ the same V-cycle is no longer enough on its own, the near-kernel of
//! $dif$ needs the auxiliary-space smoother of [`crate::hx`], but the cycle is
//! the same object at every grade, and the tower builds it uniformly through
//! [`RefinementTower::grade_vcycle`], which the auxiliary-space preconditioner
//! reuses for its blocks.
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
use multialgebra::ExteriorGrade;
use regge::lengths::mesh::MeshLengthsSq;
use simplicial::{
  linalg::{CsrMatrix, Vector},
  topology::{complex::Complex, ordering::CellOrdering, refine::Subdivision},
};

use crate::{
  linalg::DirectInverse,
  whitney_complex::{HilbertComplex, WhitneyComplex},
};

/// A refinement tower of Whitney complexes: a base mesh and `refinements`
/// successive uniform subdivisions of it, coarse to fine, with the intrinsic
/// geometry carried on every level.
///
/// It is the intrinsic (metric, not coordinate) backbone the multigrid V-cycle
/// and the auxiliary-space preconditioner both run on: it holds the complexes,
/// their [`MeshLengthsSq`] geometries and the [`Subdivision`]s linking successive
/// levels, and builds a grade-$k$ V-cycle over the whole tower on demand through
/// [`grade_vcycle`](Self::grade_vcycle). No coordinates: the tower is a source of
/// operators and transfers, both of which are metric facts, so an embedding never
/// enters here (invariant 2).
pub struct RefinementTower {
  complexes: Vec<Complex>,
  geometries: Vec<MeshLengthsSq>,
  /// `subdivisions[d]` links level `d` (coarse) to level `d + 1` (fine).
  subdivisions: Vec<Subdivision>,
}

impl RefinementTower {
  /// Build the tower by refining `base_topology`/`base_geometry` `refinements`
  /// times, halving the mesh each step.
  ///
  /// The base ordering is colex. Each refined level inherits the ordering the
  /// [`Subdivision`] carries, so the tower composes (invariant 7). Refinement is
  /// metric-free and exact, a flat cell subdivided stays flat, so the tower
  /// introduces no geometric error of its own.
  pub fn new(base_topology: Complex, base_geometry: MeshLengthsSq, refinements: usize) -> Self {
    let mut complexes = vec![base_topology];
    let mut geometries = vec![base_geometry];
    let mut subdivisions = Vec::new();
    let mut ordering = CellOrdering::colex(&complexes[0]);

    for _ in 0..refinements {
      let coarse = complexes.last().unwrap();
      let sub = coarse.refine_with(&ordering, 2);
      let fine_geometry = geometries.last().unwrap().refine(&sub, coarse);
      ordering = sub.ordering().clone();
      complexes.push(sub.complex().clone());
      geometries.push(fine_geometry);
      subdivisions.push(sub);
    }

    Self {
      complexes,
      geometries,
      subdivisions,
    }
  }

  /// The number of levels, base plus refinements.
  pub fn levels(&self) -> usize {
    self.complexes.len()
  }

  /// The index of the finest level.
  pub fn finest(&self) -> usize {
    self.complexes.len() - 1
  }

  /// The Whitney complex on level `level`.
  pub fn whitney(&self, level: usize) -> WhitneyComplex<'_> {
    WhitneyComplex::new(&self.complexes[level], &self.geometries[level])
  }

  /// The Whitney complex on the finest level, for assembling a right-hand side.
  pub fn finest_whitney(&self) -> WhitneyComplex<'_> {
    self.whitney(self.finest())
  }

  /// The subdivisions linking successive levels, exposed so an embedding can be
  /// refined alongside the tower where the extrinsic frame is needed
  /// ([`crate::hx`]).
  pub fn subdivisions(&self) -> &[Subdivision] {
    &self.subdivisions
  }

  /// The grade-`grade` multigrid V-cycle over the whole tower, with `sweeps` pre-
  /// and post-smoothing steps: reassembled operators, Whitney prolongation
  /// transfers, a damped-Jacobi smoother and a direct coarse solve.
  ///
  /// # Panics
  /// If a level's operator is not positive definite (a non-Riemannian geometry),
  /// which the direct coarse solve requires.
  pub fn grade_vcycle(
    &self,
    grade: impl Into<ExteriorGrade>,
    sweeps: usize,
  ) -> VCycle<Jacobi, DirectInverse> {
    let grade = grade.into();
    let operators: Vec<CsrMatrix> = (0..self.levels())
      .map(|l| self.whitney(l).hdif_gram(grade))
      .collect();

    // Levels finest first: for each fine level f, the transfer is the grade-k
    // Whitney prolongation of subdivisions[f - 1] and its transpose.
    let levels: Vec<Level<Jacobi>> = (1..operators.len())
      .rev()
      .map(|f| {
        let prolong = prolongation_matrix(grade, &self.complexes[f - 1], &self.subdivisions[f - 1]);
        let smoother = Jacobi::smoother(&operators[f]);
        Level::new(operators[f].clone(), smoother, prolong)
      })
      .collect();

    let coarse =
      DirectInverse::try_new(operators[0].clone()).expect("coarsest operator must be SPD");
    VCycle::new(levels, coarse, sweeps)
  }
}

/// A grade-0 multigrid solver built on a refinement tower.
///
/// Owns the [`RefinementTower`] so a right-hand side can be assembled on the
/// finest level, the finest operator, and the grade-0 V-cycle preconditioner.
/// [`solve`](Self::solve) runs V-cycle-preconditioned CG on the finest level. It
/// is the grade-0 specialization of the tower's general
/// [`grade_vcycle`](RefinementTower::grade_vcycle), where a plain V-cycle already
/// suffices.
pub struct Grade0Multigrid {
  tower: RefinementTower,
  fine_operator: CsrMatrix,
  cycle: VCycle<Jacobi, DirectInverse>,
}

impl Grade0Multigrid {
  /// Build the tower by refining `base_topology`/`base_geometry` `refinements`
  /// times (halving the mesh each step), and assemble the grade-0 V-cycle over it
  /// with `sweeps` pre- and post-smoothing steps.
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
    let tower = RefinementTower::new(base_topology, base_geometry, refinements);
    let fine_operator = tower.finest_whitney().hdif_gram(0);
    let cycle = tower.grade_vcycle(0, sweeps);
    Self {
      tower,
      fine_operator,
      cycle,
    }
  }

  /// The finest Whitney complex, for assembling a right-hand side.
  pub fn fine_complex(&self) -> WhitneyComplex<'_> {
    self.tower.finest_whitney()
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
  use regge::mesher::cartesian::CartesianGrid;

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
}
