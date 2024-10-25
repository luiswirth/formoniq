//! Solves a manufactored poisson problem in d dimensions
//! and determines the algebraic convergence rate.

#[path = "common.rs"]
mod common;

extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use common::solve_manufactured_poisson;
use formoniq::{fe::l2_norm, mesh::hyperbox::HyperBoxMeshInfo, space::FeSpace};

use std::rc::Rc;

fn main() {
  tracing_subscriber::fmt::init();

  // Spatial dimension of the problem.
  let dim: usize = 2;

  println!("Poisson in {dim}D");

  let kstart = 0;
  let kend = 10;
  let klen = kend - kstart + 1;

  // Define analytic solution.
  // $u = exp(x_1 x_2 dots x_n)$
  let analytic_sol = |x: na::DVectorView<f64>| (x.iter().product::<f64>()).exp();
  let analytic_laplacian = |x: na::DVectorView<f64>| {
    let mut prefactor = 0.0;

    for i in 0..dim {
      let mut partial_product = 1.0;
      for j in 0..dim {
        if i != j {
          partial_product *= x[j].powi(2);
        }
      }
      prefactor += partial_product;
    }

    prefactor * analytic_sol(x)
  };

  fn print_seperator() {
    let nchar = 78;
    println!("{}", "-".repeat(nchar));
  }

  print_seperator();
  println!(
    "| {:>2} | {:>13} | {:>10} | {:>16} | {:>9} | {:>9} |",
    "k", "nsubdivisions", "mesh width", "shape regularity", "L2 error", "conv rate"
  );
  print_seperator();

  let mut errors = Vec::with_capacity(klen);
  for k in kstart..=kend {
    let expk = 2usize.pow(k as u32);
    let nboxes_per_dim = expk;

    // Create mesh of unit hypercube $[0, 1]^d$.
    let box_mesh = HyperBoxMeshInfo::new_unit(dim, nboxes_per_dim);
    let coord_mesh = box_mesh.compute_coord_manifold();
    let nodes = coord_mesh.node_coords().clone();
    let mesh = Rc::new(coord_mesh.into_manifold());
    let mesh_width = mesh.mesh_width();
    let shape_regularity = mesh.shape_regularity_measure();

    // Create FE space.
    let space = FeSpace::new(mesh.clone());

    // Compute Galerkin solution to manufactored poisson problem.
    let galsol = solve_manufactured_poisson(&nodes, &space, analytic_sol, analytic_laplacian);

    // Compute analytical solution on mesh nodes.
    let analytical_sol = na::DVector::from_iterator(
      nodes.nnodes(),
      nodes.coords().column_iter().map(analytic_sol),
    );

    // Compute L2 error and convergence rate.
    let error = l2_norm(analytical_sol - galsol, &mesh);
    let conv_rate = if let Some(&prev_error) = errors.last() {
      let quot: f64 = error / prev_error;
      -quot.log2()
    } else {
      f64::INFINITY
    };
    errors.push(error);

    println!(
      "| {:>2} | {:>13} | {:>10.3e} | {:>16.3e} | {:>9.3e} | {:>9.2} |",
      k, nboxes_per_dim, mesh_width, shape_regularity, error, conv_rate
    );
  }
  print_seperator();
}
