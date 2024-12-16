//! Solves a manufactored poisson problem in d dimensions
//! and determines the algebraic convergence rate.

extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use formoniq::{fe::l2_norm, problems::poisson};
use manifold_complex::{gen::cartesian::CartesianMesh, RiemannianComplex};

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
        let coord_mesh = box_mesh.compute_coord_manifold();

        // $u = sin(x_1) + sin(x_1) + ... + sin(x_d)$
        let anal_sol = |x: na::DVectorView<f64>| x.iter().map(|x| x.sin()).sum();
        let anal_lapl = |x: na::DVectorView<f64>| x.iter().map(|x| x.sin()).sum();

        let anal_sol = coord_mesh.coords().eval_coord_fn(anal_sol);
        let anal_lapl = coord_mesh.coords().eval_coord_fn(anal_lapl);

        let (mesh, _) = coord_mesh.into_riemannian_complex();

        PoissonWithSol {
          mesh,
          solution_exact: anal_sol,
          load_data: anal_lapl,
        }
      })
      .collect();

    measure_convergence(setups);
  }
}

struct PoissonWithSol {
  mesh: RiemannianComplex,
  load_data: na::DVector<f64>,
  solution_exact: na::DVector<f64>,
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
      mesh,
      load_data,
      solution_exact,
    } = setup;

    let boundary_data = |ivertex| solution_exact[ivertex];
    let galsol = poisson::solve_poisson(&mesh, load_data, boundary_data);

    // Compute L2 error and convergence rate.
    let error = l2_norm(solution_exact - galsol, &mesh);
    let conv_rate = if let Some(&prev_error) = errors.last() {
      let quot: f64 = error / prev_error;
      -quot.log2()
    } else {
      f64::INFINITY
    };
    errors.push(error);

    let mesh_width = mesh.mesh_width_max();
    let shape_regularity = mesh.shape_regularity_measure();

    println!(
      "| {:>2} | {:>10.3e} | {:>10.3e} | {:>9.3e} | {:>9.2} |",
      refinement_level, mesh_width, shape_regularity, error, conv_rate
    );
  }
  print_seperator();
}
