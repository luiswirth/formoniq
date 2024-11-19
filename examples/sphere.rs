extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use formoniq::{
  assemble, fe,
  matrix::FaerCholesky,
  mesh::{dim3::TriangleSurface3D, sphere_surface::mesh_sphere_surface, util::NodeData},
  space::FeSpace,
};

use std::rc::Rc;

fn main() {
  let sphere_mesh = mesh_sphere_surface(5);
  let coord_mesh = sphere_mesh.into_coord_manifold();

  let mut surface = TriangleSurface3D::from_coord_manifold(coord_mesh.clone());
  std::fs::write("out/sphere_mesh.obj", surface.to_obj_string().as_bytes()).unwrap();

  let mesh = Rc::new(coord_mesh.clone().into_manifold());
  let space = FeSpace::new(mesh.clone());

  let elmat = fe::laplacian_neg_elmat;
  let galmat = assemble::assemble_galmat(&space, elmat);

  let elvec = fe::LoadElvec::new(NodeData::from_coords_map(coord_mesh.node_coords(), |p| {
    let p: na::Vector3<f64> = na::try_convert(p.into_owned()).unwrap();
    let _theta = p.z.acos(); // theta in [0,pi]
    let phi = p.y.atan2(p.x); // phi in [0,tau]
    phi.sin()
  }));
  let galvec = assemble::assemble_galvec(&space, elvec);

  let galmat = galmat.to_nalgebra_csc();
  let galsol = FaerCholesky::new(galmat).solve(&galvec);

  surface.displace_normal(galsol.as_slice());
  std::fs::write("out/sphere_sol.obj", surface.to_obj_string().as_bytes()).unwrap();
}
