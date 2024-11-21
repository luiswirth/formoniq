extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use formoniq::{
  assemble,
  fe::{laplacian_neg_elmat, lumped_mass_elmat, LoadElvec},
  lse,
  matrix::FaerCholesky,
  mesh::dim3,
  space::FeSpace,
};

#[allow(unused_imports)]
use std::f64::consts::{PI, TAU};

use std::rc::Rc;

fn main() {
  tracing_subscriber::fmt::init();

  let dim: usize = 2;

  let surface = dim3::mesh_sphere_surface(7);
  std::fs::write("out/wave_sphere_mesh.obj", surface.to_obj_string()).unwrap();

  let coord_mesh = surface.clone().into_coord_manifold();
  let mesh = Rc::new(coord_mesh.clone().into_manifold());

  let final_time = 2.0 * TAU;

  // TODO: fix CFL condition. Proper calculation for any mesh!
  // CFL condition (with 20% margin)
  let mut timestep = 0.80 * mesh.mesh_width() / (dim as f64).sqrt();
  let mut nsteps = (final_time / timestep).ceil() as usize;

  let anim_fps = 30;
  let anim_duration = 5;
  let anim_nframes = anim_fps * anim_duration;
  if nsteps < anim_nframes {
    nsteps = anim_nframes;
    timestep = final_time / nsteps as f64;
  }

  let space = FeSpace::new(Rc::clone(&mesh));

  let mut galmat_laplacian = assemble::assemble_galmat(&space, laplacian_neg_elmat);
  let mut galmat_mass = assemble::assemble_galmat(&space, lumped_mass_elmat);

  let load = na::DVector::zeros(mesh.nnodes());
  let mut galvec = assemble::assemble_galvec(&space, LoadElvec::new(load));

  assert!(!mesh.has_boundary());
  lse::enforce_homogeneous_dirichlet_bc(&mesh, &mut galmat_laplacian, &mut galvec);
  lse::enforce_homogeneous_dirichlet_bc(&mesh, &mut galmat_mass, &mut galvec);

  let galmat_laplacian = galmat_laplacian.to_nalgebra_csc();
  let galmat_mass = galmat_mass.to_nalgebra_csc();

  let galmat_mass_cholesky = FaerCholesky::new(galmat_mass.clone());

  let mut mu = coord_mesh.node_coords().eval_coord_fn(|p| {
    let p: na::Vector3<f64> = na::try_convert(p.into_owned()).unwrap();
    #[allow(unused_variables)]
    let [r, theta, phi] = dim3::cartesian2spherical(p);
    (-theta.powi(2) * 100.0).exp()
  });
  let mut nu = na::DVector::zeros(space.ndofs());

  let mut times = Vec::new();
  let mut mdd_frames: Vec<Vec<[f32; 3]>> = Vec::new();

  let last_step = nsteps - 1;
  for istep in 0..nsteps {
    println!("Step={istep}/{last_step}...");

    let t = istep as f64 / (nsteps - 1) as f64 * final_time;
    times.push(t as f32);

    let rhs = &galmat_mass * nu + timestep * (&galvec - &galmat_laplacian * &mu);
    nu = galmat_mass_cholesky.solve(&rhs);
    mu += timestep * &nu;

    //util::save_vector(&mu, format!("out/wavesol_step{istep}.txt")).unwrap();

    let mut surface = surface.clone();
    surface.displace_normal(&mu);

    let frame_coords: Vec<[f32; 3]> = surface
      .node_coords()
      .column_iter()
      .map(|col| [col.x as f32, col.y as f32, col.z as f32])
      .collect();

    mdd_frames.push(frame_coords);
  }

  println!("Writing animation into `.mdd` file.");
  dim3::write_mdd_file("out/wave_sphere_sol.mdd", &mdd_frames, &times).unwrap();
}
