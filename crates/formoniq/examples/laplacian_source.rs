extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use formoniq::problems::hodge_laplace;
use manifold::gen::cartesian::CartesianMesh;

fn main() {
  let dim = 2;
  let nboxes_per_dim = 1;
  let box_mesh = CartesianMesh::new_unit(dim, nboxes_per_dim);
  let coord_mesh = box_mesh.compute_coord_manifold();
  let (mesh, _) = coord_mesh.into_riemannian_complex();

  let form_rank = 1;
  let source = na::DVector::from_element(mesh.skeleton(form_rank).len(), 1.0);

  let (_sigma, u) = hodge_laplace::solve_hodge_laplace_source(&mesh, form_rank, source);
  println!("{u}");
}
