//! Solving Helmholtz equation.

extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use formoniq::problems::laplace_beltrami;
use manifold::gen::{cartesian::CartesianMesh, dim3::TriangleSurface3D};

fn main() {
  let dim = 2;
  let nboxes_per_dim = 100;
  let box_mesh = CartesianMesh::new_unit(dim, nboxes_per_dim);
  let coord_mesh = box_mesh.compute_coord_manifold();

  let surface = TriangleSurface3D::from_coord_manifold(coord_mesh.clone().embed_euclidean(3));
  std::fs::write("out/helmholtz_mesh.obj", surface.to_obj_string().as_bytes()).unwrap();
  let (mesh, _) = coord_mesh.into_riemannian_complex();

  let spectrum = laplace_beltrami::solve_laplace_beltrami_evp(&mesh);
  for (eigenval, eigenfunc) in spectrum.0.iter().zip(spectrum.1.column_iter()) {
    println!("eigenval={eigenval}");
    assert!((eigenval - eigenval.round()).abs() <= 10e-12);
    let eigenval = eigenval.round();

    let mut graph = surface.clone();
    let displacements = 10.0 * eigenfunc;
    graph.displace_normal(&displacements);

    std::fs::write(
      format!("out/helmholtz{eigenval}.obj"),
      graph.to_obj_string().as_bytes(),
    )
    .unwrap();
  }
}
