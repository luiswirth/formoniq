extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use formoniq::{
  assemble::{assemble_galmat_lagrangian, assemble_galvec, fix_dof_coeffs},
  fe::{laplacian_neg_elmat, lumped_mass_elmat, LoadElvec},
  mesh::factory::{linear_idx2cartesian_idx, unit_hypercube_mesh},
  space::FeSpace,
};

use std::{
  f64::consts::{PI, TAU},
  rc::Rc,
};

fn main() {
  tracing_subscriber::fmt::init();

  let d: usize = 2;
  let nsubdivisions = 100;
  let h = 1.0 / nsubdivisions as f64;

  let tfinal = 1.0 / 2.0f64.sqrt();
  let nsteps = 300;
  let tau = tfinal / nsteps as f64;

  assert!(tau < h, "CFL condition must be fullfilled.");

  let mesh = Rc::new(unit_hypercube_mesh(d, nsubdivisions));
  let nnodes = mesh.nnodes();
  let nodes_per_dim = (nnodes as f64).powf((d as f64).recip()) as usize;

  let space = Rc::new(FeSpace::new(mesh.clone()));

  let mut galmat_laplacian = assemble_galmat_lagrangian(&space, laplacian_neg_elmat);
  let mut galmat_mass = assemble_galmat_lagrangian(&space, lumped_mass_elmat);
  let mut galvec = assemble_galvec(&space, LoadElvec::new(|_| 0.0));

  let bc = |mut idof| {
    let mut fcoord = na::DVector::zeros(d);
    let mut is_boundary = false;
    for dim in 0..d {
      let icoord = idof % nodes_per_dim;
      fcoord[dim] = icoord as f64 / (nodes_per_dim - 1) as f64;
      is_boundary |= icoord == 0 || icoord == nodes_per_dim - 1;
      idof /= nodes_per_dim;
    }
    is_boundary.then_some(0.0)
  };

  fix_dof_coeffs(bc, &mut galmat_laplacian, &mut galvec);
  fix_dof_coeffs(bc, &mut galmat_mass, &mut galvec);

  let galmat_mass_cholesky = nas::factorization::CscCholesky::factor(&galmat_mass).unwrap();

  let mut file = std::fs::File::create("out/wavesol.txt").unwrap();
  std::io::Write::write_all(
    &mut file,
    format!("{d} {nodes_per_dim} {tfinal} {nsteps}\n").as_bytes(),
  )
  .unwrap();

  let mut mu = na::DVector::zeros(nnodes);
  let mut nu = na::DVector::zeros(nnodes);

  for inode in 0..nnodes {
    let x = linear_idx2cartesian_idx(inode, d, nodes_per_dim).cast::<f64>() / nodes_per_dim as f64;
    mu[inode] = (TAU * x[0]).sin() * (TAU * x[1]).sin();
  }

  let contents: String = mu.iter().map(|v| format!("{v}\n")).collect();
  std::io::Write::write_all(&mut file, contents.as_bytes()).unwrap();

  for istep in 1..nsteps {
    let t = istep as f64 / (nsteps - 1) as f64 * tfinal;
    println!("step={istep}, t={t:.2}");

    let rhs = &galmat_mass * nu + tau * (&galvec - &galmat_laplacian * &mu);
    nu = galmat_mass_cholesky.solve(&rhs).column(0).into();
    mu = mu + tau * &nu;

    let contents: String = mu.iter().map(|v| format!("{v}\n")).collect();
    std::io::Write::write_all(&mut file, contents.as_bytes()).unwrap();
  }
}
