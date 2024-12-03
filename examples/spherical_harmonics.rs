extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use formoniq::{evp, fe, mesh::dim3::mesh_sphere_surface};

use std::rc::Rc;

fn main() {
  let triangle_mesh = mesh_sphere_surface(6);

  std::fs::write(
    "out/sphere_mesh.obj",
    triangle_mesh.to_obj_string().as_bytes(),
  )
  .unwrap();

  let coord_mesh = triangle_mesh.clone().into_coord_manifold();
  let mesh = Rc::new(coord_mesh.into_manifold());

  let spectrum = evp::solve_homogeneous_evp(&mesh, fe::laplace_beltrami_elmat);
  for (eigenval, eigenfunc) in spectrum.0.iter().zip(spectrum.1.column_iter()) {
    println!("eigenval={eigenval}");
    assert!((eigenval - eigenval.round()).abs() <= 10e-12);
    let eigenval = eigenval.round();

    let mut graph = triangle_mesh.clone();
    let displacements = 10.0 * eigenfunc;
    graph.displace_normal(&displacements);

    std::fs::write(
      format!("out/spherical_harmonic{eigenval}.obj"),
      graph.to_obj_string().as_bytes(),
    )
    .unwrap();
  }
}
