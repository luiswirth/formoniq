extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use formoniq::problems::laplace_beltrami;
use manifold::gen::dim3::mesh_sphere_surface;

fn main() {
  let triangle_mesh = mesh_sphere_surface(4);
  let coord_mesh = triangle_mesh.clone().into_coord_manifold();
  let (mesh, _) = coord_mesh.into_riemannian_complex();

  let spectrum = laplace_beltrami::solve_laplace_beltrami_evp(&mesh);
  for (i, (&eigenval, eigenfunc)) in spectrum.0.iter().zip(spectrum.1.column_iter()).enumerate() {
    println!("eigenval={eigenval:.2}");
    let eigenfunc = eigenval * eigenfunc.normalize();

    let mut surface = triangle_mesh.clone();
    surface.displace_normal(&eigenfunc);
    std::fs::write(
      format!("out/spectrum{i}.obj"),
      surface.to_obj_string().as_bytes(),
    )
    .unwrap();
  }
}
