extern crate nalgebra as na;

use formoniq::{
  assemble::assemble_galmat, fe::laplacian_neg_elmat, mesh::factory, space::FeSpace,
};

use std::rc::Rc;

fn main() {
  tracing_subscriber::fmt::init();

  let ncells = 3;
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
  let mesh = factory::from_facets(nodes, cells, false);
  let mesh = Rc::new(mesh);
  let space = Rc::new(FeSpace::new(mesh));
  let galmat = assemble_galmat(&space, laplacian_neg_elmat);
  let galmat = na::DMatrix::from(&galmat);
  println!("{galmat}");
}
