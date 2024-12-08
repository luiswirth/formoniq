//! Solving Helmholtz equation.

extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use formoniq::{
  evp, fe,
  mesh::{dim3, hyperbox::HyperBoxMeshInfo},
};

use std::rc::Rc;

fn main() {
  let dim = 2;
  let nboxes_per_dim = 100;
  let box_mesh = HyperBoxMeshInfo::new_unit(dim, nboxes_per_dim);
  let coord_mesh = box_mesh.to_coord_manifold();

  let surface = dim3::TriangleSurface3D::from_coord_manifold(coord_mesh.clone().embed_flat(3));
  std::fs::write("out/helmholtz_mesh.obj", surface.to_obj_string().as_bytes()).unwrap();
  let mesh = Rc::new(coord_mesh.into_manifold());

  let spectrum = evp::solve_homogeneous_evp(&mesh, fe::laplace_beltrami_elmat);
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
