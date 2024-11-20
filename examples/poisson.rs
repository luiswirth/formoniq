//! Solves a manufactored poisson problem in d dimensions
//! and determines the algebraic convergence rate.

extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use formoniq::{
  fe::l2_norm,
  mesh::{hyperbox::HyperBoxMeshInfo, SimplicialManifold},
  solve_poisson,
};

use std::{f64::consts::TAU, rc::Rc};

fn main() {
  tracing_subscriber::fmt::init();

  let dim = 2;

  let setups = (0..8)
    .map(|refinement| {
      let nboxes_per_dim = 2usize.pow(refinement);

      // Mesh of hypercube $[0, tau]^d$.
      let box_mesh = HyperBoxMeshInfo::new_unit_scaled(dim, nboxes_per_dim, TAU);
      let coord_mesh = box_mesh.to_coord_manifold();

      // $u = sin(x_1) + sin(x_1) + ... + sin(x_d)$
      let anal_sol = |x: na::DVectorView<f64>| x.iter().map(|x| x.sin()).sum();
      let anal_lapl = |x: na::DVectorView<f64>| x.iter().map(|x| x.sin()).sum();

      let anal_sol = coord_mesh.node_coords().eval_coord_fn(anal_sol);
      let anal_lapl = coord_mesh.node_coords().eval_coord_fn(anal_lapl);

      let mesh = Rc::new(coord_mesh.into_manifold());

      PoissonWithSol {
        mesh,
        solution_exact: anal_sol,
        load_data: anal_lapl,
      }
    })
    .collect();

  measure_convergence(setups);
}

struct PoissonWithSol {
  mesh: Rc<SimplicialManifold>,
  load_data: na::DVector<f64>,
  solution_exact: na::DVector<f64>,
}

/// Supply analytic solution and analytic (negative) Laplacian
fn measure_convergence(refined_setups: Vec<PoissonWithSol>) {
  fn print_seperator() {
    let nchar = 78;
    println!("{}", "-".repeat(nchar));
  }

  print_seperator();
  println!(
    "| {:>2} | {:>10} | {:>16} | {:>9} | {:>9} |",
    "k", "mesh width", "shape regularity", "L2 error", "conv rate"
  );
  print_seperator();

  let mut errors = Vec::with_capacity(refined_setups.len());
  for (refinement_level, setup) in refined_setups.into_iter().enumerate() {
    let PoissonWithSol {
      mesh,
      load_data,
      solution_exact,
    } = setup;

    let boundary_data = |inode| solution_exact[inode];
    let galsol = solve_poisson(&mesh, load_data, boundary_data);

    // Compute L2 error and convergence rate.
    let error = l2_norm(solution_exact - galsol, &mesh);
    let conv_rate = if let Some(&prev_error) = errors.last() {
      let quot: f64 = error / prev_error;
      -quot.log2()
    } else {
      f64::INFINITY
    };
    errors.push(error);

    let mesh_width = mesh.mesh_width();
    let shape_regularity = mesh.shape_regularity_measure();

    println!(
      "| {:>2} | {:>10.3e} | {:>16.3e} | {:>9.3e} | {:>9.2} |",
      refinement_level, mesh_width, shape_regularity, error, conv_rate
    );
  }
  print_seperator();
}
