//! Linearized Mean Curvature Flow / Evolution to Minimal Surface

use common::linalg::{
  faer::FaerCholesky,
  nalgebra::{CsrMatrix, Matrix},
};
use ddf::cochain::Cochain;
use formoniq::{bc, whitney_complex::WhitneyComplex};
use manifold::geometry::coord::mesh::MeshCoords;

#[allow(unused_imports)]
use std::f64::consts::{PI, TAU};

fn main() {
  tracing_subscriber::fmt::init();

  let obj_path = std::env::var("FORMONIQ_OBJ_PATH")
    .expect("specify the OBJ file using the envvar `FORMONIQ_OBJ_PATH`");
  let obj_string = std::fs::read_to_string(obj_path).unwrap();
  std::fs::write("out/curvature_flow.obj", &obj_string).unwrap();
  let surface = manifold::io::blender::from_obj_string(&obj_string);

  let (topology, coords) = surface.into_coord_complex();
  let mut metric = coords.to_edge_lengths(&topology);
  let nvertices = topology.vertices().len();

  let mut coords_list = vec![coords];

  let dt = 1e-3;
  let nsteps = 10;

  let last_step = nsteps - 1;
  for istep in 0..nsteps {
    println!("Solving Curvature Flow at step={istep}/{last_step}...");

    let fes = WhitneyComplex::new(&topology, &metric);
    let laplace = CsrMatrix::from(&fes.codif_dif(0));
    let mass = CsrMatrix::from(&fes.mass(0));
    let system = &mass + dt * &laplace;
    let boundary = fes.boundary();
    let relative = fes.relative();

    let coords_initial = coords_list.first().unwrap();
    let coords_old = coords_list.last().unwrap();
    let mut coords_new = Matrix::zeros(coords_initial.dim(), nvertices);
    for d in 0..coords_initial.dim() {
      let comps_initial = Cochain::new(0, coords_initial.matrix().row(d).transpose());
      let comps_old = coords_old.matrix().row(d).transpose();

      let rhs = &mass * comps_old;
      let comps_new = match &boundary {
        // Boundary held fixed at the initial position.
        Some(boundary) => {
          let boundary_values = boundary.trace_cochain(&comps_initial);
          bc::solve_with_essential_bc(&relative, boundary, system.clone(), &rhs, &boundary_values)
            .coeffs
        }
        None => FaerCholesky::new(system.clone()).solve(&rhs),
      };
      coords_new.row_mut(d).copy_from(&comps_new.transpose());
    }

    let coords_new = MeshCoords::new(coords_new);
    metric = coords_new.to_edge_lengths(&topology);
    coords_list.push(coords_new);
  }

  let coords_list: Vec<_> = coords_list.into_iter().collect();
  let times = (0..=nsteps).map(|istep| istep as f64 * dt);
  manifold::io::blender::write_3dmesh_animation(&coords_list, times);
}
