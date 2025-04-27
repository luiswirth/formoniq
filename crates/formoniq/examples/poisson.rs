//! Solves a manufactored poisson problem in d dimensions
//! and determines the algebraic convergence rate.

use {
  common::{linalg::nalgebra::Vector, util::algebraic_convergence_rate},
  ddf::cochain::cochain_projection,
  exterior::field::DiffFormClosure,
  formoniq::{
    assemble::assemble_galvec, fe::fe_l2_error, operators::SourceElVec, problems::laplace_beltrami,
  },
  manifold::gen::cartesian::CartesianMeshInfo,
};

use std::f64::consts::TAU;

fn main() {
  tracing_subscriber::fmt::init();

  for dim in 1_usize..=3 {
    println!("Solving Poisson in {dim}d.");

    // $u = sin(x_1) + sin(x_1) + ... + sin(x_d)$
    let solution_exact = DiffFormClosure::scalar(|x| x.iter().map(|x| x.sin()).sum(), dim);
    let dif_solution_exact = DiffFormClosure::one_form(
      |x| Vector::from_iterator(x.len(), x.iter().map(|xi| xi.cos())),
      dim,
    );
    let laplacian_exact = DiffFormClosure::scalar(|x| x.iter().map(|x| x.sin()).sum(), dim);

    let nrefinements = 20 / dim;
    let mut errors_l2 = Vec::with_capacity(nrefinements);
    let mut errors_h1 = Vec::with_capacity(nrefinements);

    println!(
      "| {:>2} | {:>8.2} | {:>6.2} | {:>8.2} | {:>6.2} |",
      "k", "L2 err", "L2 conv", "H1 err", "H1 conv",
    );

    for irefine in 0..nrefinements {
      let nboxes_per_dim = 2usize.pow(irefine as u32);

      // Mesh of hypercube $[0, tau]^d$.
      let box_mesh = CartesianMeshInfo::new_unit_scaled(dim, nboxes_per_dim, TAU);
      let (topology, coords) = box_mesh.compute_coord_complex();
      let metric = coords.to_edge_lengths(&topology);

      let load_vector = assemble_galvec(
        &topology,
        &metric,
        SourceElVec::new(&laplacian_exact, &coords, None),
      );

      let solution_projected = cochain_projection(&solution_exact, &topology, &coords, None);
      let boundary_data = |ivertex| solution_projected[ivertex];

      let galsol = laplace_beltrami::solve_laplace_beltrami_source(
        &topology,
        &metric,
        load_vector,
        boundary_data,
      );

      let conv_rate = |errors: &[f64], curr: f64| {
        errors
          .last()
          .map(|&prev| algebraic_convergence_rate(curr, prev))
          .unwrap_or(f64::INFINITY)
      };

      let error_l2 = fe_l2_error(&galsol, &solution_exact, &topology, &coords);
      let conv_rate_l2 = conv_rate(&errors_l2, error_l2);
      errors_l2.push(error_l2);

      let dif_galsol = galsol.dif(&topology);
      let error_h1 = fe_l2_error(&dif_galsol, &dif_solution_exact, &topology, &coords);
      let conv_rate_h1 = conv_rate(&errors_h1, error_h1);
      errors_h1.push(error_h1);

      println!(
        "| {:>2} | {:<8.2e} | {:>6.2} | {:<8.2e} | {:>6.2} |",
        irefine, error_l2, conv_rate_l2, error_h1, conv_rate_h1
      );
    }
  }
}
