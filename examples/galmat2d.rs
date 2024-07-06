extern crate nalgebra as na;

use formoniq::{assemble::assemble_galmat, fe::laplacian_neg_elmat, mesh::factory, space::FeSpace};

use std::rc::Rc;

fn main() {
  tracing_subscriber::fmt::init();

  let nodes = na::DMatrix::from_column_slice(2, 4, &[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
  let cells = vec![vec![0, 1, 2], vec![3, 2, 1]];
  let mesh = factory::from_facets(nodes, cells, false);

  let mesh = Rc::new(mesh);
  let space = Rc::new(FeSpace::new(mesh));
  let galmat = assemble_galmat(&space, laplacian_neg_elmat);
  let galmat = na::DMatrix::from(&galmat);
  println!("{galmat}");
}
