//! Linearized Mean Curvature Flow / Evolution to Minimal Surface

extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use formoniq::{
  assemble, fe,
  mesh::{coordinates::VertexCoords, gen::dim3},
  util::FaerCholesky,
};

#[allow(unused_imports)]
use std::f64::consts::{PI, TAU};

fn main() {
  tracing_subscriber::fmt::init();

  let obj_path = std::env::var("FORMONIQ_OBJ_PATH")
    .expect("specify the OBJ file using the envvar `FORMONIQ_OBJ_PATH`");
  let obj_string = std::fs::read_to_string(obj_path).unwrap();
  let surface = dim3::TriangleSurface3D::from_obj_string(&obj_string);
  std::fs::write("out/curvature_flow.obj", surface.to_obj_string()).unwrap();

  let coord_mesh = surface.into_coord_manifold();
  let (mut mesh, coords) = coord_mesh.into_riemannian_complex();

  let mut coords_list = vec![coords];

  let dt = 1e-3;
  let nsteps = 10;

  let last_step = nsteps - 1;
  for istep in 0..nsteps {
    println!("Solving Curvature Flow at step={istep}/{last_step}...");

    let laplace = assemble::assemble_galmat(&mesh, fe::laplace_beltrami_elmat);
    let mass = assemble::assemble_galmat(&mesh, fe::mass_elmat);
    let source = na::DVector::zeros(mesh.nvertices());

    let coords_initial = coords_list.first().unwrap();
    let coords_old = coords_list.last().unwrap();
    let mut coords_new = na::DMatrix::zeros(coords_initial.dim(), mesh.nvertices());
    for d in 0..coords_initial.dim() {
      let comps_initial = coords_initial.matrix().row(d).transpose();
      let comps_old = coords_old.matrix().row(d).transpose();

      let mut laplace = laplace.clone();
      let mut mass = mass.clone();
      let mut source = source.clone();

      let boundary_data = |inode| comps_initial[inode];
      assemble::enforce_dirichlet_bc(&mesh, boundary_data, &mut laplace, &mut source);
      assemble::enforce_dirichlet_bc(&mesh, boundary_data, &mut mass, &mut source);

      let laplace = laplace.to_nalgebra_csc();
      let mass = mass.to_nalgebra_csc();

      let lse_matrix = &mass + dt * &laplace;
      let lse_cholesky = FaerCholesky::new(lse_matrix);

      let rhs = &mass * comps_old + dt * &source;
      let comps_new = lse_cholesky.solve(&rhs).transpose();
      coords_new.row_mut(d).copy_from(&comps_new);
    }

    let coords_new = VertexCoords::new(coords_new);
    *mesh.edge_lengths_mut() = coords_new.to_edge_lengths(mesh.complex());
    coords_list.push(coords_new);
  }

  let mdd_frames: Vec<Vec<[f32; 3]>> = coords_list
    .into_iter()
    .map(|coords| {
      let frame_coords: Vec<[f32; 3]> = coords
        .matrix()
        .column_iter()
        .map(|col| [col[0] as f32, col[1] as f32, col[2] as f32])
        .collect();

      frame_coords
    })
    .collect();
  let times: Vec<_> = (0..=nsteps).map(|istep| dt as f32 * istep as f32).collect();

  println!("Writing animation to `.mdd` file...");
  dim3::write_mdd_file("out/curvature_flow.mdd", &mdd_frames, &times).unwrap();
}
