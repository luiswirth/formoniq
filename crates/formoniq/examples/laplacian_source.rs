#![allow(warnings)]

extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use std::ops::Neg;

use exterior::{dense::KForm, manifold::discretize_mesh};
use formoniq::problems::hodge_laplace;
use geometry::coord::{manifold::cartesian::CartesianMesh, Coord};

fn main() {
  let dim = 2;
  let nboxes_per_dim = 10;
  let box_mesh = CartesianMesh::new_unit(dim, nboxes_per_dim);
  let coord_mesh = box_mesh.compute_coord_manifold().into_coord_complex();
  let (mesh, _) = coord_mesh.clone().into_metric_complex();

  let form_rank = 1;
  let source = |x: Coord| KForm::new(x, dim, form_rank);

  let source = discretize_mesh(&source, form_rank, &coord_mesh).coeffs;

  let (_sigma, u) = hodge_laplace::solve_hodge_laplace_source(&mesh, form_rank, source);
  println!("{u}");
}
