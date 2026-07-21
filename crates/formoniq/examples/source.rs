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
  derham::section::CoordFieldExt,
  formoniq::{
    assemble::assemble_galvec, fe::fe_l2_error, operators::SourceElVec, problems::elliptic,
    whitney_complex::WhitneyComplex,
  },
  simplicial::{mesher::cartesian::CartesianGrid, topology::ordering::CellOrdering},
  util::{BoundaryCondition, BoxEigenform, algebraic_convergence_rate, report},
};

use std::f64::consts::PI;

fn main() {
  tracing_subscriber::fmt::init();

  const BCS: [BoundaryCondition; 2] = [BoundaryCondition::Absolute, BoundaryCondition::Relative];

  for dim in 1..=3 {
    for grade in 0..=dim {
      let forms = BCS.map(|bc| BoxEigenform::new(dim, grade, bc));

      // Whenever the harmonic space is nontrivial, fixing the harmonic part
      // runs a sparse shift-invert Lanczos solve for exactly the harmonic
      // dimension's worth of eigenpairs, so the budget here is wall-clock
      // patience for a hand-run sweep, not a dense O(N^3) ceiling.
      const MAX_DOFS: usize = 20_000;

      // Rows are gathered per BC and printed as grouped tables afterwards,
      // so the mesh and Whitney complex — independent of `bc` — are built
      // once per refinement and shared by both BCs, not duplicated.
      let mut errors_l2 = [const { Vec::new() }; 2];
      let mut errors_hd = [const { Vec::new() }; 2];
      let mut rows = [const { Vec::new() }; 2];

      // The h-family is a Freudenthal refinement tower over one coarse box,
      // rather than a grid regenerated at each resolution: the meshes are nested,
      // the coarse Whitney space a subspace of the fine one, and the path is
      // mesh-agnostic. Refining a flat cell is exact, so no geometric error
      // enters the rate.
      //
      // Each level refines the previous one, carrying the ordering the
      // subdivision inherits: that is what makes refinement compose, so the tower
      // stays inside the Kuhn family and every cell remains similar to the coarse
      // box. Without the ordering the levels would drift out of the family above
      // dimension two, degrading the shape constants.
      let (mut topology, mut coords) = CartesianGrid::new_unit_scaled(dim, 1, PI).triangulate();
      let mut metric = coords.to_edge_lengths_sq(&topology);
      let mut ordering = CellOrdering::colex(&topology);
      for irefine in 0u32..=8 {
        if irefine > 0 {
          let sub = topology.refine_with(&ordering, 2);
          // Intrinsic and extrinsic refinement side by side: the metric is
          // transported through the Regge primitive, the embedding follows only
          // because the analytic fields are pulled back through it.
          metric = metric.refine(&sub, &topology);
          coords = coords.refine(&sub);
          ordering = sub.ordering().clone();
          topology = sub.into_complex();
        }
        let whitney = WhitneyComplex::new(&topology, &metric);

        let ndofs = whitney.ndofs(grade)
          + if grade > 0 {
            whitney.ndofs(grade - 1)
          } else {
            0
          };
        if !errors_l2[0].is_empty() && ndofs > MAX_DOFS {
          break;
        }

        let ncells = topology.cells().len();
        for (i, bc) in BCS.into_iter().enumerate() {
          let form = &forms[i];

          // The continuum eigenform becomes a field on the mesh by pullback along
          // the affine cell charts; everything downstream is intrinsic.
          let (solution_field, load_field) = (form.solution(), form.load());
          let solution = solution_field.pullback_on(&topology, &coords);
          let load = load_field.pullback_on(&topology, &coords);

          let source = assemble_galvec(&topology, &metric, SourceElVec::new(&load, None));
          // Absolute BC is the full Whitney complex, relative BC its subcomplex of
          // boundary-vanishing cochains; the solver is one piece of code over both.
          let (_, galsol, _) = match bc {
            BoundaryCondition::Absolute => elliptic::solve_source(&whitney, source, grade).unwrap(),
            BoundaryCondition::Relative => {
              elliptic::solve_source(&whitney.relative(), source, grade).unwrap()
            }
          };

          let conv = |errors: &[f64], curr: f64| {
            errors.last().map_or(f64::INFINITY, |&prev| {
              algebraic_convergence_rate(curr, prev)
            })
          };

          let error_l2 = fe_l2_error(&galsol, &solution, &topology, &metric);
          let conv_l2 = conv(&errors_l2[i], error_l2);
          errors_l2[i].push(error_l2);

          // Top grade: $dif u = 0$ identically, so there is no $H(dif)$ error to
          // converge — the two columns are marked absent rather than filled with a
          // spurious zero.
          let (error_hd, conv_hd) = match form.dif_solution() {
            Some(dif_solution_field) => {
              let dif_solution = dif_solution_field.pullback_on(&topology, &coords);
              let error_hd = fe_l2_error(&galsol.dif(&topology), &dif_solution, &topology, &metric);
              let conv_hd = conv(&errors_hd[i], error_hd);
              errors_hd[i].push(error_hd);
              (Some(error_hd), Some(conv_hd))
            }
            None => (None, None),
          };

          rows[i].push(format!(
            "| {irefine:>2} | {ncells:>7} | {:>9} | {:>7} | {:>9} | {:>7} |",
            report::err(Some(error_l2)),
            report::rate(Some(conv_l2)),
            report::err(error_hd),
            report::rate(conv_hd),
          ));
        }
      }

      for (i, bc) in BCS.into_iter().enumerate() {
        println!(
          "\nHodge-Laplace source — dim {dim}, grade {grade}, {} — Δu = {}u",
          bc.label(),
          forms[i].eigenvalue()
        );
        println!(
          "| {:>2} | {:>7} | {:>9} | {:>7} | {:>9} | {:>7} |",
          "r", "ncells", "L2 err", "L2 conv", "Hd err", "Hd conv",
        );
        for row in &rows[i] {
          println!("{row}");
        }
      }
    }
  }
}
