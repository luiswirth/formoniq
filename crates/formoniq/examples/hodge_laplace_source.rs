//! Hodge-Laplace source problem on the flat box $[0, pi]^n$, swept over every
//! dimension $1 <= n <= 3$ and every form grade $0 <= k <= n$.
//!
//! One loop realizes the whole classical zoo — Poisson at $k = 0$, the vector
//! Laplacian at $k = 1$, the top-form problem at $k = n$ — as instances of the
//! single mixed Hodge-Laplace problem, on the single dimension-general Cartesian
//! mesh. The manufactured solution is likewise generic: the natural-BC eigenform
//! with $Delta u = n u$ (see [`util::BoxEigenform`]), so the load is $n u$ and the
//! source problem's exact solution is an eigenform.
//!
//! The sweep starts at $n = 1$: at $n = 0$ the box is a point, $Delta$ is the
//! zero operator, every form is harmonic, and the mixed formulation returns the
//! non-harmonic part — which is $0$. There is no non-harmonic solution to
//! manufacture, so the source problem is degenerate. The eigenvalue example does
//! run $n = 0$, where the spectrum ${0}$ is still meaningful.
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
  util::BoxEigenform,
};

use std::f64::consts::PI;

fn main() {
  tracing_subscriber::fmt::init();

  for dim in 1..=3 {
    for grade in 0..=dim {
      let form = BoxEigenform::new(dim, grade);
      println!(
        "\nHodge-Laplace source, dim {dim}, grade {grade} (Delta u = {} u).",
        form.eigenvalue()
      );
      println!(
        "| {:>2} | {:>7} | {:>8} | {:>7} | {:>8} | {:>7} |",
        "r", "ncells", "L2 err", "L2 conv", "Hd err", "Hd conv",
      );

      // At grade 0 the harmonic projection needs the constant harmonic form,
      // extracted with the dense generalized eigensolver, so the refinement
      // budget is set by what that $O(N^3)$ solve tolerates, not the sparse LU.
      let max_refine = match dim {
        1 => 8,
        2 => 5,
        _ => 3,
      };

      let mut errors_l2 = Vec::new();
      let mut errors_hd = Vec::new();
      for irefine in 0..=max_refine {
        let nboxes_per_dim = 2usize.pow(irefine);
        let box_mesh = CartesianMeshInfo::new_unit_scaled(dim, nboxes_per_dim, PI);
        let (topology, coords) = box_mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);

        // The continuum eigenform becomes a field on the mesh by pullback along
        // the affine cell charts; everything downstream is intrinsic.
        let (solution_field, load_field) = (form.solution(), form.load());
        let solution = solution_field.pullback_on(&topology, &coords);
        let load = load_field.pullback_on(&topology, &coords);

        let source = assemble_galvec(&topology, &metric, SourceElVec::new(&load, None));
        let (_, galsol, _) = hodge_laplace::solve_hodge_laplace_source(
          WhitneyComplex::new(&topology, &metric),
          source,
          grade,
        );

        let conv = |errors: &[f64], curr: f64| {
          errors.last().map_or(f64::INFINITY, |&prev| {
            algebraic_convergence_rate(curr, prev)
          })
        };

        let error_l2 = fe_l2_error(&galsol, &solution, &topology, &metric);
        let conv_l2 = conv(&errors_l2, error_l2);
        errors_l2.push(error_l2);

        let (error_hd, conv_hd) = match form.dif_solution() {
          Some(dif_solution_field) => {
            let dif_solution = dif_solution_field.pullback_on(&topology, &coords);
            let error_hd = fe_l2_error(&galsol.dif(&topology), &dif_solution, &topology, &metric);
            let conv_hd = conv(&errors_hd, error_hd);
            errors_hd.push(error_hd);
            (error_hd, conv_hd)
          }
          // Top grade: $dif u = 0$ identically, so the exterior derivative of the
          // discrete solution is exactly zero and there is nothing to converge.
          None => (0.0, f64::NAN),
        };

        let ncells = topology.cells().len();
        println!(
          "| {irefine:>2} | {ncells:>7} | {error_l2:<8.2e} | {conv_l2:>7.2} | {error_hd:<8.2e} | {conv_hd:>7.2} |"
        );
      }
    }
  }
}
