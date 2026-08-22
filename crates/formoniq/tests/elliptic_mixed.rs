//! The mixed Hodge-Laplace saddle point: with the $H Lambda(dif)$ block
//! preconditioner its MINRES iteration count stays bounded under refinement,
//! the Arnold-Falk-Winther norm equivalence in its operational form.

use formoniq::galerkin::GalerkinVector;
use formoniq::problems::elliptic::{
  assemble_mixed_kkt, mixed_block_preconditioner, solve_harmonics,
};
use formoniq::whitney_complex::WhitneyComplex;
use iterative::StopCriterion;
use iterative::krylov::minres;
use regge::mesher::cartesian::CartesianGrid;
use simplicial::{Dim, linalg::Vector};

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
