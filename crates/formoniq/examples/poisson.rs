//! Solves a manufactored poisson problem in d dimensions
//! and determines the algebraic convergence rate.

extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use {
  common::util::algebraic_convergence_rate,
  exterior::field::DiffFormClosure,
  formoniq::{
    fe::{h1_norm, l2_norm},
    problems::laplace_beltrami,
  },
  manifold::{
    gen::cartesian::CartesianMeshInfo, geometry::metric::MeshEdgeLengths,
    topology::complex::Complex,
  },
  whitney::cochain::{de_rham_map, Cochain},
};

use std::f64::consts::TAU;

fn main() {
  tracing_subscriber::fmt::init();

  for dim in 1_usize..=3 {
    println!("Solving Poisson in {dim}d.");

    // $u = sin(x_1) + sin(x_1) + ... + sin(x_d)$
    let solution_exact = DiffFormClosure::scalar(|x| x.iter().map(|x| x.sin()).sum(), dim);
    let laplacian = DiffFormClosure::scalar(|x| x.iter().map(|x| x.sin()).sum(), dim);

    let nrefinements = 20 / dim;
    let setups = (0..nrefinements)
      .map(move |refinement| {
        let nboxes_per_dim = 2usize.pow(refinement as u32);

        // Mesh of hypercube $[0, tau]^d$.
        let box_mesh = CartesianMeshInfo::new_unit_scaled(dim, nboxes_per_dim, TAU);
        let (topology, coords) = box_mesh.compute_coord_complex();

        let solution_exact = de_rham_map(&solution_exact, &topology, &coords);
        let laplacian = de_rham_map(&laplacian, &topology, &coords);

        let metric = coords.to_edge_lengths(&topology);

        PoissonWithSol {
          topology,
          metric,
          solution_exact,
          load_data: laplacian,
        }
      })
      .collect();

    measure_convergence(setups);
  }
}

struct PoissonWithSol {
  topology: Complex,
  metric: MeshEdgeLengths,
  load_data: Cochain,
  solution_exact: Cochain,
}

/// Supply analytic solution and analytic (negative) Laplacian
fn measure_convergence(refined_setups: Vec<PoissonWithSol>) {
  fn print_seperator() {
    let nchar = 24;
    println!("{}", "-".repeat(nchar));
  }

  print_seperator();
  println!("| {:>2} | {:>6} | {:>6} |", "k", "L2", "H1",);
  print_seperator();

  let mut errors_l2 = Vec::with_capacity(refined_setups.len());
  let mut errors_h1 = Vec::with_capacity(refined_setups.len());
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

    let difference = solution_exact - galsol;

    let conv_rate = |errors: &[f64], curr: f64| {
      errors
        .last()
        .map(|&prev| algebraic_convergence_rate(curr, prev))
        .unwrap_or(f64::INFINITY)
    };

    let error_l2 = l2_norm(&difference, &topology, &metric);
    let conv_rate_l2 = conv_rate(&errors_l2, error_l2);
    errors_l2.push(error_l2);

    let error_h1 = h1_norm(&difference, &topology, &metric);
    let conv_rate_h1 = conv_rate(&errors_h1, error_h1);
    errors_h1.push(error_h1);

    println!(
      "| {:>2} | {:>6.2} | {:>6.2} |",
      refinement_level, conv_rate_l2, conv_rate_h1,
    );
  }
  print_seperator();
}
