extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use std::f64::consts::PI;

use formoniq::problems::hodge_laplace;
use geometry::coord::manifold::{cartesian::CartesianMesh, dim3::TriangleSurface3D};

fn main() {
  let dim = 2;
  let nboxes_per_dim = 30;
  let box_mesh = CartesianMesh::new_unit(dim, nboxes_per_dim);
  let coord_mesh = box_mesh.compute_coord_manifold();
  let surface = TriangleSurface3D::from_coord_skeleton(coord_mesh.clone());
  let (mesh, _) = coord_mesh.into_metric_complex();

  let neigen_values = 10;
  let form_rank = 2;
  let spectrum = hodge_laplace::solve_hodge_laplace_evp(&mesh, form_rank, neigen_values);
  for (&eigenval, eigenfunc) in spectrum.0.iter().zip(spectrum.1.column_iter()) {
    let eigenval_reduced = (eigenval / PI.powi(2)).round() as u32;
    println!("eigenval={eigenval}, eigenval'={eigenval_reduced}");

    let mut surface = surface.clone();
    let displacement = eigenval * eigenfunc.normalize();
    surface.displace_normal(&displacement);
    std::fs::write(
      format!("out/spectrum{eigenval_reduced}.obj"),
      surface.to_obj_string().as_bytes(),
    )
    .unwrap();
  }
}
