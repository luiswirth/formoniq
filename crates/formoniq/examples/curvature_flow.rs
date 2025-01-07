//! Linearized Mean Curvature Flow / Evolution to Minimal Surface

extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use common::util::FaerCholesky;
use formoniq::{assemble, fe};
use geometry::coord::{
  manifold::dim3::{write_3dmesh_animation, TriangleSurface3D},
  VertexCoords,
};

#[allow(unused_imports)]
use std::f64::consts::{PI, TAU};

fn main() {
  tracing_subscriber::fmt::init();

  let obj_path = std::env::var("FORMONIQ_OBJ_PATH")
    .expect("specify the OBJ file using the envvar `FORMONIQ_OBJ_PATH`");
  let obj_string = std::fs::read_to_string(obj_path).unwrap();
  let surface = TriangleSurface3D::from_obj_string(&obj_string);
  std::fs::write("out/curvature_flow.obj", surface.to_obj_string()).unwrap();

  let (mut mesh, coords) = surface.into_coord_skeleton().into_metric_complex();
  let nvertices = mesh.topology().vertices().len();

  let mut coords_list = vec![coords];

  let dt = 1e-3;
  let nsteps = 10;

  let last_step = nsteps - 1;
  for istep in 0..nsteps {
    println!("Solving Curvature Flow at step={istep}/{last_step}...");

    let laplace = assemble::assemble_galmat(&mesh, fe::LaplaceBeltramiElmat);
    let mass = assemble::assemble_galmat(&mesh, fe::ScalarMassElmat);
    let source = na::DVector::zeros(nvertices);

    let coords_initial = coords_list.first().unwrap();
    let coords_old = coords_list.last().unwrap();
    let mut coords_new = na::DMatrix::zeros(coords_initial.dim(), nvertices);
    for d in 0..coords_initial.dim() {
      let comps_initial = coords_initial.matrix().row(d).transpose();
      let comps_old = coords_old.matrix().row(d).transpose();

      let mut laplace = laplace.clone();
      let mut mass = mass.clone();
      let mut source = source.clone();

      let boundary_data = |inode| comps_initial[inode];
      assemble::enforce_dirichlet_bc(mesh.topology(), boundary_data, &mut laplace, &mut source);
      assemble::enforce_dirichlet_bc(mesh.topology(), boundary_data, &mut mass, &mut source);

      let laplace = laplace.to_nalgebra_csr();
      let mass = mass.to_nalgebra_csr();

      let lse_matrix = &mass + dt * &laplace;
      let lse_cholesky = FaerCholesky::new(lse_matrix);

      let rhs = &mass * comps_old + dt * &source;
      let comps_new = lse_cholesky.solve(&rhs).transpose();
      coords_new.row_mut(d).copy_from(&comps_new);
    }

    let coords_new = VertexCoords::new(coords_new);
    *mesh.edge_lengths_mut() = coords_new.to_edge_lengths(
      mesh
        .topology()
        .edges()
        .iter()
        .map(|e| e.simplex_set().clone().try_into().unwrap()),
    );
    coords_list.push(coords_new);
  }

  let coords_list: Vec<_> = coords_list
    .into_iter()
    .map(|coords| coords.into_const_dim())
    .collect();
  let times = (0..=nsteps).map(|istep| istep as f64 * dt);
  write_3dmesh_animation(&coords_list, times);
}
