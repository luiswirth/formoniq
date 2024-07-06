extern crate nalgebra as na;

use formoniq::{
  assemble::assemble_galmat, fe::laplacian_neg_elmat, mesh::factory, space::FeSpace,
};

use std::rc::Rc;

fn main() {
  tracing_subscriber::fmt::init();

  let nodes = na::DMatrix::from_column_slice(
    3,
    8,
    &[
      0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0,
      0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ],
  );
  let cells = vec![
    vec![0, 1, 3, 7],
    vec![0, 1, 5, 7],
    vec![0, 2, 3, 7],
    vec![0, 2, 6, 7],
    vec![0, 4, 5, 7],
    vec![0, 4, 6, 7],
  ];
  let mesh = factory::from_facets(nodes, cells, false);

  let mesh = Rc::new(mesh);
  let space = Rc::new(FeSpace::new(mesh));
  let galmat = assemble_galmat(&space, laplacian_neg_elmat);
  let galmat = na::DMatrix::from(&galmat);
  println!("{galmat:.2}");
}
