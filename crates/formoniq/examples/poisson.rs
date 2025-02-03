//! Solves a manufactored poisson problem in d dimensions
//! and determines the algebraic convergence rate.

extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use formoniq::{fe::l2_norm, operators::FeFunction, problems::laplace_beltrami};
use geometry::{coord::manifold::cartesian::CartesianMesh, metric::MeshEdgeLengths};
use topology::complex::TopologyComplex;

use std::f64::consts::TAU;

fn main() {
  tracing_subscriber::fmt::init();

  for dim in 1_usize..=3 {
    println!("Solving Poisson in {dim}d.");
    let nrefinements = 20 / dim;
    let setups = (0..nrefinements)
      .map(|refinement| {
        let nboxes_per_dim = 2usize.pow(refinement as u32);

        // Mesh of hypercube $[0, tau]^d$.
        let box_mesh = CartesianMesh::new_unit_scaled(dim, nboxes_per_dim, TAU);
        let (topology, coords) = box_mesh.compute_coord_complex();

        // $u = sin(x_1) + sin(x_1) + ... + sin(x_d)$
        let anal_sol = |x: na::DVectorView<f64>| x.iter().map(|x| x.sin()).sum();
        let anal_lapl = |x: na::DVectorView<f64>| x.iter().map(|x| x.sin()).sum();

        let anal_sol = coords.coord_iter().map(anal_sol).collect::<Vec<_>>().into();
        let anal_sol = FeFunction::new(0, anal_sol);

        let anal_lapl = coords
          .coord_iter()
          .map(anal_lapl)
          .collect::<Vec<_>>()
          .into();
        let anal_lapl = FeFunction::new(0, anal_lapl);

        let metric = coords.to_edge_lengths(&topology);

        PoissonWithSol {
          topology,
          metric,
          solution_exact: anal_sol,
          load_data: anal_lapl,
        }
      })
      .collect();

    measure_convergence(setups);
  }
}

struct PoissonWithSol {
  topology: TopologyComplex,
  metric: MeshEdgeLengths,
  load_data: FeFunction,
  solution_exact: FeFunction,
}

/// Supply analytic solution and analytic (negative) Laplacian
fn measure_convergence(refined_setups: Vec<PoissonWithSol>) {
  fn print_seperator() {
    let nchar = 56;
    println!("{}", "-".repeat(nchar));
  }

  print_seperator();
  println!(
    "| {:>2} | {:>10} | {:>10} | {:>9} | {:>9} |",
    "k", "mesh width", "regularity", "L2 error", "conv rate"
  );
  print_seperator();

  let mut errors = Vec::with_capacity(refined_setups.len());
  for (refinement_level, setup) in refined_setups.into_iter().enumerate() {
    let PoissonWithSol {
      topology,
      metric,
      load_data,
      solution_exact,
    } = setup;

    let boundary_data = |ivertex| solution_exact[ivertex];
    let galsol =
      laplace_beltrami::solve_laplace_beltrami_source(&topology, &metric, load_data, boundary_data);

    // Compute L2 error and convergence rate.
    let error = l2_norm(&(solution_exact - galsol), &topology, &metric);
    let conv_rate = if let Some(&prev_error) = errors.last() {
      let quot: f64 = error / prev_error;
      -quot.log2()
    } else {
      f64::INFINITY
    };
    errors.push(error);

    let mesh_width = metric.mesh_width_max();
    let shape_regularity = metric.shape_regularity_measure(&topology);

    println!(
      "| {:>2} | {:>10.3e} | {:>10.3e} | {:>9.3e} | {:>9.2} |",
      refinement_level, mesh_width, shape_regularity, error, conv_rate
    );
  }
  print_seperator();
}
