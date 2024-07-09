extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use formoniq::{
  assemble::{assemble_galmat, assemble_galvec, fix_dof_coeffs},
  fe::{laplacian_neg_elmat, lumped_mass_elmat, LoadElvec},
  matrix::FaerCholesky,
  mesh::factory::{hypercube_mesh, linear_idx2cartesian_coords},
  space::FeSpace,
};

use std::{f64::consts::TAU, rc::Rc};

// $u(x, t) = sin(x_1) sin(x_2) ... sin(x_d) cos(sqrt(d) t)$

fn main() {
  tracing_subscriber::fmt::init();

  let ndims: usize = 2;
  let nsubdivisions = 500;
  let domain_length = TAU;
  let mesh_width = domain_length / nsubdivisions as f64;

  let final_time = TAU / (ndims as f64).sqrt();

  // CFL condition (with 5% margin)
  let mut timestep = 0.95 * mesh_width / (ndims as f64).sqrt();
  let mut nsteps = (final_time / timestep).ceil() as usize;

  let anim_fps = 30;
  let anim_duration = 5;
  let anim_nframes = anim_fps * anim_duration;
  if nsteps < anim_nframes {
    nsteps = anim_nframes;
    timestep = final_time as f64 / nsteps as f64;
  }

  let mesh = hypercube_mesh(ndims, nsubdivisions, domain_length);
  let mesh = Rc::new(mesh);
  let nnodes = mesh.nnodes();
  let nodes_per_dim = (nnodes as f64).powf((ndims as f64).recip()) as usize;

  let space = Rc::new(FeSpace::new(mesh.clone()));

  let mut galmat_laplacian = assemble_galmat(&space, laplacian_neg_elmat);
  let mut galmat_mass = assemble_galmat(&space, lumped_mass_elmat);
  let mut galvec = assemble_galvec(&space, LoadElvec::new(|_| 0.0));

  let bc = |mut idof| {
    let mut fcoord = na::DVector::zeros(ndims);
    let mut is_boundary = false;
    for dim in 0..ndims {
      let icoord = idof % nodes_per_dim;
      fcoord[dim] = icoord as f64 / (nodes_per_dim - 1) as f64;
      is_boundary |= icoord == 0 || icoord == nodes_per_dim - 1;
      idof /= nodes_per_dim;
    }
    is_boundary.then_some(0.0)
  };

  fix_dof_coeffs(bc, &mut galmat_laplacian, &mut galvec);
  fix_dof_coeffs(bc, &mut galmat_mass, &mut galvec);

  let galmat_laplacian = galmat_laplacian.to_nalgebra();
  let galmat_mass = galmat_mass.to_nalgebra();

  let galmat_mass_cholesky = FaerCholesky::new(galmat_mass.clone());

  let mut file = std::fs::File::create("out/wavesol.txt").unwrap();
  std::io::Write::write_all(
    &mut file,
    format!("{ndims} {domain_length} {nodes_per_dim} {final_time} {nsteps}\n").as_bytes(),
  )
  .unwrap();

  let mut mu = na::DVector::from_iterator(
    nnodes,
    (0..nnodes).map(|inode| {
      let x = linear_idx2cartesian_coords(inode, ndims, nodes_per_dim, domain_length);
      (0..ndims).map(|idim| x[idim].sin()).product()
    }),
  );
  let mut nu = na::DVector::zeros(nnodes);

  for istep in 0..nsteps {
    let _t = istep as f64 / (nsteps - 1) as f64 * final_time;
    println!("Solving wave equation at step={istep}/{nsteps}...");

    let rhs = &galmat_mass * nu + timestep * (&galvec - &galmat_laplacian * &mu);
    nu = galmat_mass_cholesky.solve(&rhs).column(0).into();
    mu = mu + timestep * &nu;

    let contents: String = mu.iter().map(|v| format!("{v}\n")).collect();
    std::io::Write::write_all(&mut file, contents.as_bytes()).unwrap();
  }
}
