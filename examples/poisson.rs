extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use formoniq::{
  assemble::{assemble_galmat_lagrangian, fix_dof_coeffs},
  mesh::factory,
  space::FeSpace,
};

use std::rc::Rc;

fn main() {
  tracing_subscriber::fmt::init();

  let ncells = 1;
  let nnodes = ncells + 1;
  let nodes: Vec<_> = (0..nnodes).map(|c| c as f64).collect();
  let nodes = na::DMatrix::from_column_slice(1, nodes.len(), &nodes);
  let cells: Vec<_> = (0..nnodes)
    .collect::<Vec<_>>()
    .windows(2)
    .map(|w| w.to_vec())
    .collect();
  assert!(cells.len() == ncells);
  println!("{cells:?}");
  let mesh = factory::from_facets(nodes, cells);
  let mesh = Rc::new(mesh);
  let space = Rc::new(FeSpace::new(mesh));
  let mut galmat = assemble_galmat_lagrangian(space);
  let mut galvec = na::DVector::zeros(galmat.ncols());
  fix_dof_coeffs(|idof| (idof == 0).then_some(0.0), &mut galmat, &mut galvec);
  let mu = nas::factorization::CscCholesky::factor(&galmat)
    .unwrap()
    .solve(&galvec);
  println!("{mu}");
}
