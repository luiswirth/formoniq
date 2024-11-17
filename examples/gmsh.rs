//! Solves a manufactored poisson on a gmsh.

#[path = "common.rs"]
mod common;

extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use common::solve_manufactured_poisson;
use formoniq::{fe::l2_norm, mesh::gmsh, space::FeSpace};

use std::rc::Rc;

fn main() {
  tracing_subscriber::fmt::init();

  let gmsh = std::fs::read("/home/luis/untitled.msh").unwrap();
  let coord_mesh = gmsh::gmsh2coord_mesh(&gmsh);
  let nodes = coord_mesh.node_coords().clone();
  let mesh = Rc::new(coord_mesh.into_manifold());
  let dim = mesh.dim();
  let mesh_width = mesh.mesh_width();
  let shape_regularity = mesh.shape_regularity_measure();
  println!("h={mesh_width}");
  println!("rho={shape_regularity}");

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
  println!("error={error}");
}
