extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use formoniq::{
  assemble,
  fe::{laplacian_neg_elmat, lumped_mass_elmat, LoadElvec},
  lse,
  matrix::FaerCholesky,
  mesh::{dim3, hyperbox::HyperBoxMeshInfo},
  space::FeSpace,
};

use std::{f64::consts::TAU, rc::Rc};

// $u(x, t) = sin(x_1) sin(x_2) ... sin(x_d) cos(sqrt(d) t)$

fn main() {
  tracing_subscriber::fmt::init();

  let dim: usize = 2;
  let nboxes_per_dim = 50;
  let domain_length = TAU;
  let mesh_width = domain_length / nboxes_per_dim as f64;

  let final_time = TAU / (dim as f64).sqrt();

  // CFL condition (with 5% margin)
  let mut timestep = 0.95 * mesh_width / (dim as f64).sqrt();
  let mut nsteps = (final_time / timestep).ceil() as usize;

  let anim_fps = 30;
  let anim_duration = 5;
  let anim_nframes = anim_fps * anim_duration;
  if nsteps < anim_nframes {
    nsteps = anim_nframes;
    timestep = final_time / nsteps as f64;
  }

  let box_mesh = HyperBoxMeshInfo::new_unit_scaled(dim, nboxes_per_dim, domain_length);
  let coord_mesh = box_mesh.to_coord_manifold();

  let mut surface = None;
  if dim == 2 {
    let surface = surface.insert(dim3::TriangleSurface3D::from_coord_manifold(
      coord_mesh.clone().embed_flat(3),
    ));
    std::fs::write("out/wave_domain_mesh.obj", surface.to_obj_string()).unwrap();
  }

  let mesh = Rc::new(coord_mesh.clone().into_manifold());
  let space = FeSpace::new(Rc::clone(&mesh));

  let mut galmat_laplacian = assemble::assemble_galmat(&space, laplacian_neg_elmat);
  let mut galmat_mass = assemble::assemble_galmat(&space, lumped_mass_elmat);

  let load = na::DVector::zeros(mesh.nnodes());
  let mut galvec = assemble::assemble_galvec(&space, LoadElvec::new(load));

  lse::enforce_homogeneous_dirichlet_bc(&mesh, &mut galmat_laplacian, &mut galvec);
  lse::enforce_homogeneous_dirichlet_bc(&mesh, &mut galmat_mass, &mut galvec);

  let galmat_laplacian = galmat_laplacian.to_nalgebra_csc();
  let galmat_mass = galmat_mass.to_nalgebra_csc();

  let galmat_mass_cholesky = FaerCholesky::new(galmat_mass.clone());

  let mut mu = coord_mesh
    .node_coords()
    .eval_coord_fn(|p| (0..dim).map(|idim| p[idim].sin()).product());
  let mut nu = na::DVector::zeros(space.ndofs());

  let mut times = Vec::new();
  let mut mdd_frames: Vec<Vec<[f32; 3]>> = Vec::new();

  for istep in 0..nsteps {
    println!("Step={istep}/{nsteps}...");

    let t = istep as f64 / (nsteps - 1) as f64 * final_time;
    times.push(t as f32);

    let rhs = &galmat_mass * nu + timestep * (&galvec - &galmat_laplacian * &mu);
    nu = galmat_mass_cholesky.solve(&rhs);
    mu += timestep * &nu;

    //util::save_vector(&mu, format!("out/wavesol_step{istep}.txt")).unwrap();

    if dim == 2 {
      let mut surface = surface.as_ref().unwrap().clone();
      surface.displace_normal(&mu);

      let frame_coords: Vec<[f32; 3]> = surface
        .node_coords()
        .column_iter()
        .map(|col| [col.x as f32, col.y as f32, col.z as f32])
        .collect();

      mdd_frames.push(frame_coords);
    }
  }

  dim3::write_mdd_file("out/wavesol_anim.mdd", &mdd_frames, &times).unwrap();
}
