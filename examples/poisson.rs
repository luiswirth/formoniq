//! Solves a manufactored poisson problem in d dimensions
//! and determines the algebraic convergence rate.

extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use formoniq::{
  assemble,
  fe::{self, l2_norm},
  matrix::FaerCholesky,
  mesh::{hyperbox::HyperBoxMeshInfo, util::NodeData},
  space::FeSpace,
  Dim,
};

use std::rc::Rc;

fn main() {
  tracing_subscriber::fmt::init();

  let dim = 3;

  // $u = sin(x_1) + sin(x_1) + ... + sin(x_d)$
  let anal_sol = |x: na::DVectorView<f64>| x.iter().map(|x| x.sin()).sum();
  let anal_lapl = |x: na::DVectorView<f64>| anal_sol(x);

  manufactured_poisson_convergence(dim, &anal_sol, &anal_lapl);
}

fn manufactured_poisson_convergence<F, G>(
  dim: Dim,
  anal_sol: &F,
  // negative Laplacian
  anal_lapl: &G,
) where
  F: Fn(na::DVectorView<f64>) -> f64,
  G: Fn(na::DVectorView<f64>) -> f64,
{
  println!("Poisson in {dim}D");

  let kstart = 0;
  let kend = 8;
  let klen = kend - kstart + 1;

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

    // Assemble Galerkin matrix.
    let elmat = fe::laplacian_neg_elmat;
    let mut galmat = assemble::assemble_galmat(&space, elmat);

    // Assemble Galerkin vector.
    let elvec = fe::LoadElvec::new(NodeData::from_coords_map(&nodes, anal_lapl));
    let mut galvec = assemble::assemble_galvec(&space, elvec);

    // Enforce Dirichlet boundary conditions by fixing dofs on boundary.
    assemble::fix_dof_coeffs(
      |idof| {
        box_mesh.is_node_on_boundary(idof).then(|| {
          let dof_pos = box_mesh.node_pos(idof);
          anal_sol(dof_pos.as_view())
        })
      },
      &mut galmat,
      &mut galvec,
    );

    let galmat = galmat.to_nalgebra_csc();

    // Obtain Galerkin solution by solving LSE.
    let galsol: na::DVector<f64> = FaerCholesky::new(galmat).solve(&galvec).column(0).into();

    // Compute analytical solution on mesh nodes.
    let analytical_sol =
      na::DVector::from_iterator(nodes.nnodes(), nodes.coords().column_iter().map(anal_sol));

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
