extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use formoniq::{
  assemble::{self, assemble_galmat_lagrangian, assemble_galvec},
  mesh::factory,
  space::FeSpace,
};

use std::rc::Rc;

fn main() {
  tracing_subscriber::fmt::init();

  // Create mesh of unit iterval [0, 1].
  let ncells = 1000;
  let nnodes = ncells + 1;
  let nodes: Vec<_> = (0..nnodes)
    .map(|c| c as f64 / (nnodes - 1) as f64)
    .collect();
  let nodes = na::DMatrix::from_column_slice(1, nodes.len(), &nodes);
  let cells: Vec<_> = (0..nnodes)
    .collect::<Vec<_>>()
    .windows(2)
    .map(|w| w.to_vec())
    .collect();
  assert!(cells.len() == ncells);
  let mesh = factory::from_facets(nodes.clone(), cells);
  let mesh = Rc::new(mesh);

  // Create FE space.
  let space = Rc::new(FeSpace::new(mesh));
  let ndofs = space.ndofs();

  // Assemble galerkin matrix and galerkin vector.
  let mut galmat = assemble_galmat_lagrangian(space.clone());
  let mut galvec = assemble_galvec(space, |_| 1.0);

  // Enforce homogeneous dirichlet boundary conditions
  // by dropping dofs on boundary.
  assemble::drop_dofs(
    |idof| idof == 0 || idof == ndofs - 1,
    &mut galmat,
    &mut galvec,
  );

  // Obtain galerkin solution by solving LSE.
  let galsol = nas::factorization::CscCholesky::factor(&galmat)
    .unwrap()
    .solve(&galvec);

  // Compute exact analytical solution on nodes.
  let exact_sol = |x: f64| 0.5 * x - 0.5 * x * x;
  let exact_sol = nodes.map(exact_sol).transpose();
  let exact_sol = exact_sol.view_range(1..(ndofs - 1), ..);

  // Compute error norm.
  let error = exact_sol - galsol;
  let error = error.norm();
  println!("error = {:e}", error);
}
