extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use formoniq::problems::laplace_beltrami;
use manifold::gen::dim3::mesh_sphere_surface;

fn main() {
  let triangle_mesh = mesh_sphere_surface(6);
  let coord_mesh = triangle_mesh.clone().into_coord_manifold();
  let (mesh, _) = coord_mesh.into_riemannian_complex();

  let spectrum = laplace_beltrami::solve_laplace_beltrami_evp(&mesh, 10);

  for (i, (&eigenval, eigenfunc)) in spectrum.0.iter().zip(spectrum.1.column_iter()).enumerate() {
    println!("eigenval={eigenval:.2}");

    let mut surface = triangle_mesh.clone();
    for (ivertex, mut cart_pos) in surface.vertex_coords_mut().coord_iter_mut().enumerate() {
      cart_pos *= eigenfunc[ivertex];
    }
    std::fs::write(
      format!("out/spherical_harmonic{i}.obj"),
      surface.to_obj_string().as_bytes(),
    )
    .unwrap();
  }
}
