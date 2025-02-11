extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use formoniq::{
  operators::FeFunction,
  problems::wave::{self, WaveState},
};
use manifold::dim3::{cartesian2spherical, mesh_sphere_surface};

#[allow(unused_imports)]
use std::f64::consts::{PI, TAU};

fn main() {
  tracing_subscriber::fmt::init();

  println!("Meshing sphere...");
  let mesh_subdivisions = 5;
  let surface = mesh_sphere_surface(mesh_subdivisions);
  println!("Writing mesh to `.obj` file...");
  std::fs::write("out/sphere_wave.obj", manifold::io::to_obj_string(&surface)).unwrap();

  let (topology, coords) = surface.clone().into_coord_complex();
  let metric = coords.to_edge_lengths(&topology);
  let nvertices = topology.vertices().len();

  let final_time = 2.0 * TAU;
  // TODO: fix CFL conditions!!!
  let mut dt = 0.3 * wave::cfl_dt(&metric, 1.0);
  let mut nsteps = (final_time / dt) as usize;

  let anim_time = 5;
  let min_anim_fps = 30;
  let min_anim_nsteps = min_anim_fps * anim_time;
  if nsteps < min_anim_nsteps {
    nsteps = min_anim_nsteps;
    dt = final_time / nsteps as f64;
  }

  let times: Vec<_> = (0..=nsteps).map(|istep| istep as f64 * dt).collect();

  assert!(!topology.has_boundary());
  let boundary_data = |_| unreachable!();
  let force_data = FeFunction::zero(0, &topology);

  let initial_pos = coords
    .coord_iter()
    .map(|p| {
      let p: na::Vector3<f64> = na::try_convert(p.into_owned()).unwrap();
      #[allow(unused_variables)]
      let [r, theta, phi] = cartesian2spherical(p);

      (-theta.powi(2) * 50.0).exp()
    })
    .collect::<Vec<_>>()
    .into();
  let initial_vel = na::DVector::zeros(nvertices);
  let initial_data = WaveState::new(initial_pos, initial_vel);

  let solution = wave::solve_wave(
    &topology,
    &metric,
    &times,
    boundary_data,
    initial_data,
    force_data,
  );

  let displacements: Vec<_> = solution.into_iter().map(|s| s.pos).collect();
  manifold::io::write_displacement_animation(&surface, &displacements, times.iter().copied());
}
