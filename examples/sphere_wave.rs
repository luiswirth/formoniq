extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use formoniq::{
  mesh::dim3,
  wave::{self, cfl_dt, WaveState},
};

use std::rc::Rc;

#[allow(unused_imports)]
use std::f64::consts::{PI, TAU};

fn main() {
  tracing_subscriber::fmt::init();

  println!("Meshing sphere...");
  let mesh_subdivisions = 5;
  let surface = dim3::mesh_sphere_surface(mesh_subdivisions);
  println!("Writing mesh to `.obj` file...");
  std::fs::write("out/sphere_wave.obj", surface.to_obj_string()).unwrap();

  let coord_mesh = surface.clone().into_coord_manifold();
  let mesh = Rc::new(coord_mesh.clone().into_manifold());

  let final_time = 2.0 * TAU;
  // TODO: fix CFL conditions!!!
  let mut dt = 0.3 * cfl_dt(&mesh, 1.0);
  let mut nsteps = (final_time / dt) as usize;

  let anim_time = 5;
  let min_anim_fps = 30;
  let min_anim_nsteps = min_anim_fps * anim_time;
  if nsteps < min_anim_nsteps {
    nsteps = min_anim_nsteps;
    dt = final_time / nsteps as f64;
  }

  let times: Vec<_> = (0..=nsteps).map(|istep| istep as f64 * dt).collect();

  assert!(!mesh.has_boundary());
  let boundary_data = |_| unreachable!();

  let initial_pos = coord_mesh.node_coords().eval_coord_fn(|p| {
    let p: na::Vector3<f64> = na::try_convert(p.into_owned()).unwrap();
    #[allow(unused_variables)]
    let [r, theta, phi] = dim3::cartesian2spherical(p);

    (-theta.powi(2) * 50.0).exp()
  });
  let initial_vel = na::DVector::zeros(mesh.nnodes());
  let initial_data = WaveState::new(initial_pos, initial_vel);

  let force_data = na::DVector::zeros(mesh.nnodes());

  let solution = wave::solve_wave(&mesh, &times, boundary_data, initial_data, force_data);

  let mdd_frames: Vec<Vec<[f32; 3]>> = solution
    .into_iter()
    .map(|state| {
      let mut surface = surface.clone();
      surface.displace_normal(&state.pos);

      let frame_coords: Vec<[f32; 3]> = surface
        .node_coords()
        .column_iter()
        .map(|col| [col.x as f32, col.y as f32, col.z as f32])
        .collect();

      frame_coords
    })
    .collect();
  let times: Vec<_> = times.into_iter().map(|f| f as f32).collect();

  println!("Writing animation to `.mdd` file...");
  dim3::write_mdd_file("out/sphere_wave.mdd", &mdd_frames, &times).unwrap();
}
