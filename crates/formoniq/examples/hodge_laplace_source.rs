//! Hodge-Laplace source problem on the flat box $[0, pi]^n$, swept over every
//! dimension $1 <= n <= 3$, every form grade $0 <= k <= n$, and both boundary
//! conditions.
//!
//! One loop realizes the whole classical zoo — Poisson at $k = 0$, the vector
//! Laplacian at $k = 1$, the top-form problem at $k = n$ — as instances of the
//! single mixed Hodge-Laplace problem, on the single dimension-general Cartesian
//! mesh. Each runs under both the absolute (natural / Neumann) and the relative
//! (essential / Dirichlet) boundary conditions, the two Hodge duals: the same
//! solver on the full Whitney complex and on its boundary-vanishing subcomplex.
//! The manufactured solution is likewise generic: the eigenform with $Delta u =
//! n u$ (see [`util::BoxEigenform`]), the two BCs its $sin arrow.l.r cos$ swap,
//! so the load is $n u$ and the source problem's exact solution is an eigenform.
//!
//! The sweep starts at $n = 1$: at $n = 0$ the box is a point, $Delta$ is the
//! zero operator, every form is harmonic, and the only compatible load is $0$
//! with unique non-harmonic solution $0$. The degeneracy is intrinsic — no
//! source data escapes $Delta equiv 0$ — not an artifact of the manufactured
//! eigenform, so a point genuinely has no source problem. The base case lives in
//! the eigenvalue example instead, where the spectrum ${0}$ is still meaningful.
//!
//! Run by hand; read the convergence rates off the tables. Whitney forms are
//! first order, so expect $O(h)$ in both $L^2$ and $H(dif)$.

#[path = "util/mod.rs"]
mod util;

use {
  common::util::algebraic_convergence_rate,
  ddf::section::CoordFieldExt,
  formoniq::{
    assemble::assemble_galvec, fe::fe_l2_error, operators::SourceElVec, problems::hodge_laplace,
    whitney_complex::WhitneyComplex,
  },
  manifold::gen::cartesian::CartesianMeshInfo,
  util::{report, BoundaryCondition, BoxEigenform},
};

use std::f64::consts::PI;

fn main() {
  tracing_subscriber::fmt::init();

  for dim in 1..=3 {
    for grade in 0..=dim {
      for bc in [BoundaryCondition::Absolute, BoundaryCondition::Relative] {
        let form = BoxEigenform::new(dim, grade, bc);
        println!(
          "\nHodge-Laplace source — dim {dim}, grade {grade}, {} — Δu = {}u",
          bc.label(),
          form.eigenvalue()
        );
        println!(
          "| {:>2} | {:>7} | {:>9} | {:>7} | {:>9} | {:>7} |",
          "r", "ncells", "L2 err", "L2 conv", "Hd err", "Hd conv",
        );

        // Whenever the harmonic space is nontrivial, fixing the harmonic part
        // runs a dense generalized eigensolve, $O(N^3)$ in the mixed-system size
        // $N = dim cal(W) Lambda^k + dim cal(W) Lambda^(k-1)$ at this grade. That
        // cost, not the dimension, sets the refinement budget: absolute hits it
        // at grade $0$ (smallest system), relative at top grade (largest), so a
        // per-dimension cap would misjudge both. Stop at the first level over
        // budget. This is the same rule the eigenvalue example uses.
        const MAX_DOFS: usize = 1200;

        let mut errors_l2 = Vec::new();
        let mut errors_hd = Vec::new();
        for irefine in 0u32..=8 {
          let nboxes_per_dim = 2usize.pow(irefine);
          let box_mesh = CartesianMeshInfo::new_unit_scaled(dim, nboxes_per_dim, PI);
          let (topology, coords) = box_mesh.compute_coord_complex();
          let metric = coords.to_edge_lengths(&topology);
          let whitney = WhitneyComplex::new(&topology, &metric);

          let ndofs = whitney.ndofs(grade)
            + if grade > 0 {
              whitney.ndofs(grade - 1)
            } else {
              0
            };
          if !errors_l2.is_empty() && ndofs > MAX_DOFS {
            break;
          }

          // The continuum eigenform becomes a field on the mesh by pullback along
          // the affine cell charts; everything downstream is intrinsic.
          let (solution_field, load_field) = (form.solution(), form.load());
          let solution = solution_field.pullback_on(&topology, &coords);
          let load = load_field.pullback_on(&topology, &coords);

          let source = assemble_galvec(&topology, &metric, SourceElVec::new(&load, None));
          // Absolute BC is the full Whitney complex, relative BC its subcomplex of
          // boundary-vanishing cochains; the solver is one piece of code over both.
          let (_, galsol, _) = match bc {
            BoundaryCondition::Absolute => {
              hodge_laplace::solve_hodge_laplace_source(&whitney, source, grade)
            }
            BoundaryCondition::Relative => {
              hodge_laplace::solve_hodge_laplace_source(&whitney.relative(), source, grade)
            }
          };

          let conv = |errors: &[f64], curr: f64| {
            errors.last().map_or(f64::INFINITY, |&prev| {
              algebraic_convergence_rate(curr, prev)
            })
          };

          let error_l2 = fe_l2_error(&galsol, &solution, &topology, &metric);
          let conv_l2 = conv(&errors_l2, error_l2);
          errors_l2.push(error_l2);

          // Top grade: $dif u = 0$ identically, so there is no $H(dif)$ error to
          // converge — the two columns are marked absent rather than filled with a
          // spurious zero.
          let (error_hd, conv_hd) = match form.dif_solution() {
            Some(dif_solution_field) => {
              let dif_solution = dif_solution_field.pullback_on(&topology, &coords);
              let error_hd = fe_l2_error(&galsol.dif(&topology), &dif_solution, &topology, &metric);
              let conv_hd = conv(&errors_hd, error_hd);
              errors_hd.push(error_hd);
              (Some(error_hd), Some(conv_hd))
            }
            None => (None, None),
          };

          let ncells = topology.cells().len();
          println!(
            "| {irefine:>2} | {ncells:>7} | {:>9} | {:>7} | {:>9} | {:>7} |",
            report::err(Some(error_l2)),
            report::rate(Some(conv_l2)),
            report::err(error_hd),
            report::rate(conv_hd),
          );
        }
      }
    }
  }
}
