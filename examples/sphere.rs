extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use formoniq::{mesh::dim3::mesh_sphere_surface, solve_poisson};

use std::rc::Rc;

fn main() {
  let mut triangle_mesh = mesh_sphere_surface(5);

  std::fs::write(
    "out/sphere_mesh.obj",
    triangle_mesh.to_obj_string().as_bytes(),
  )
  .unwrap();

  let coord_mesh = triangle_mesh.clone().into_coord_manifold();

  let load_data = coord_mesh.node_coords().eval_coord_fn(|p| {
    let p: na::Vector3<f64> = na::try_convert(p.into_owned()).unwrap();
    let _theta = p.z.acos(); // theta in [0,pi]
    let phi = p.y.atan2(p.x); // phi in [0,tau]
    phi.sin()
  });
  let boundary_data = |_| unreachable!();

  let mesh = Rc::new(coord_mesh.into_manifold());
  let galsol = solve_poisson(&mesh, load_data, boundary_data);

  triangle_mesh.displace_normal(galsol.as_slice());
  std::fs::write(
    "out/sphere_sol.obj",
    triangle_mesh.to_obj_string().as_bytes(),
  )
  .unwrap();
}
