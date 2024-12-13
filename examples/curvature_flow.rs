//! Linearized Mean Curvature Flow / Evolution to Minimal Surface

extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use formoniq::{
  assemble, fe,
  mesh::gen::dim3::{self, TriangleSurface3D},
  util::FaerCholesky,
  Dim,
};

#[allow(unused_imports)]
use std::f64::consts::{PI, TAU};

fn main() {
  tracing_subscriber::fmt::init();

  const EXTRINSIC_DIM: Dim = 3;

  let mesh_subdivisions = 5;
  let surface = dim3::mesh_sphere_surface(mesh_subdivisions);
  std::fs::write("out/curvature_flow0.obj", surface.to_obj_string()).unwrap();

  let mut coord_mesh = surface.into_coord_manifold();
  let mut mesh = coord_mesh.to_riemannian_complex();

  let dt = 1e-3;
  let nsteps = 30;

  let last_step = nsteps - 1;
  for istep in 0..nsteps {
    println!("Solving Curvature Flow at step={istep}/{last_step}...");

    let laplace = assemble::assemble_galmat(&mesh, fe::laplace_beltrami_elmat);
    let mass = assemble::assemble_galmat(&mesh, fe::mass_elmat);
    let source = na::DVector::zeros(mesh.nvertices());

    let laplace = laplace.to_nalgebra_csc();
    let mass = mass.to_nalgebra_csc();

    let lse_matrix = &mass + dt * &laplace;
    let lse_cholesky = FaerCholesky::new(lse_matrix);

    let mut new_coords = coord_mesh.coords().clone();
    for d in 0..EXTRINSIC_DIM {
      let components = coord_mesh.coords().matrix().row(d).transpose();
      let rhs = &mass * components + dt * &source;
      let new_components = lse_cholesky.solve(&rhs);
      new_coords
        .matrix_mut()
        .row_mut(d)
        .copy_from(&new_components.transpose());
    }

    *coord_mesh.coords_mut() = new_coords;
    *mesh.edge_lengths_mut() = coord_mesh.coords().to_edge_lengths(mesh.complex());
  }

  let surface = TriangleSurface3D::from_coord_manifold(coord_mesh);
  std::fs::write("out/curvature_flow1.obj", surface.to_obj_string()).unwrap();
}
