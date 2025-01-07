#![allow(warnings)]

extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use std::ops::Neg;

use exterior::dense::KForm;
use formoniq::problems::hodge_laplace;
use geometry::coord::{manifold::cartesian::CartesianMesh, Coord};

fn main() {
  let dim = 2;
  let nboxes_per_dim = 1;
  let box_mesh = CartesianMesh::new_unit(dim, nboxes_per_dim);
  let coord_mesh = box_mesh.compute_coord_manifold();
  let (mesh, _) = coord_mesh.into_metric_complex();

  let form_rank = 1;
  let source = |x: Coord| {
    KForm::new(
      (&x - na::dvector![0.5, 0.5]).norm_squared().neg().exp() * x.normalize(),
      dim,
      form_rank,
    )
  };
  //let source = coord_mesh.integrate_form();
  let source = todo!();

  let (_sigma, u) = hodge_laplace::solve_hodge_laplace_source(&mesh, form_rank, source);
  println!("{u}");
}
