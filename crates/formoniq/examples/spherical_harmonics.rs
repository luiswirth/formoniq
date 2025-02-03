extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use formoniq::problems::laplace_beltrami;
use geometry::coord::manifold::dim3::mesh_sphere_surface;

fn main() {
  let triangle_mesh = mesh_sphere_surface(6);
  let (topology, coords) = triangle_mesh.clone().into_coord_complex();
  let metric = coords.to_edge_lengths(&topology);

  let spectrum = laplace_beltrami::solve_laplace_beltrami_evp(&topology, &metric, 10);

  for (i, (&eigenval, eigenfunc)) in spectrum.0.iter().zip(spectrum.1).enumerate() {
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
