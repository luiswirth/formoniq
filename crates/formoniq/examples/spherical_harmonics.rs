extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use std::fs;

use formoniq::problems::laplace_beltrami;
use manifold::dim3::mesh_sphere_surface;

fn main() {
  let path = "out/spherical_harmonics";
  let _ = fs::remove_dir_all(path);
  fs::create_dir_all(path).unwrap();

  let triangle_mesh = mesh_sphere_surface(6);
  let (topology, coords) = triangle_mesh.clone().into_coord_complex();
  let metric = coords.to_edge_lengths(&topology);

  let (eigenvals, eigenfuncs) =
    laplace_beltrami::solve_laplace_beltrami_evp(&topology, &metric, 10);

  for (i, (&eigenval, eigenfunc)) in eigenvals.iter().zip(eigenfuncs).enumerate() {
    println!("eigenval={eigenval:.2}");

    let mut surface = triangle_mesh.clone();
    for (ivertex, mut cart_pos) in surface.vertex_coords_mut().coord_iter_mut().enumerate() {
      cart_pos *= eigenfunc[ivertex];
    }
    std::fs::write(
      format!("{path}/harmonic{i}.obj"),
      manifold::io::blender::to_obj_string(&surface).as_bytes(),
    )
    .unwrap();
  }
}
