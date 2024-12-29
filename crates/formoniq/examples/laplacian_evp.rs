extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use std::f64::consts::PI;

use formoniq::problems::laplace_beltrami;
use manifold::gen::cartesian::CartesianMesh;

fn main() {
  let dim = 1;
  let nboxes_per_dim = 100;
  let box_mesh = CartesianMesh::new_unit(dim, nboxes_per_dim);
  let coord_mesh = box_mesh.compute_coord_manifold();
  let (mesh, _) = coord_mesh.into_riemannian_complex();

  let spectrum = laplace_beltrami::solve_laplace_beltrami_evp(&mesh);
  for (i, (eigenval, _eigenfunc)) in spectrum.0.iter().zip(spectrum.1.column_iter()).enumerate() {
    println!(
      "eigenval={eigenval}, npi^2={}",
      ((i + 1) as f64 * PI).powi(2)
    );
  }
}
