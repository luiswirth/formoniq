extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use formoniq::{
  assemble::{assemble_galmat_lagrangian, assemble_galvec, fix_dof_coeffs},
  fe::{laplacian_neg_elmat, lumped_mass_elmat, LoadElvec},
  mesh::factory::unit_hypercube_mesh,
  space::FeSpace,
};

use std::{f64::consts::TAU, rc::Rc};

fn main() {
  tracing_subscriber::fmt::init();

  let d: usize = 1;
  let nsubdivisions = 100;

  let tfinal = 1.0;
  let nsteps = 200;
  let tau = tfinal / nsteps as f64;

  assert!(nsteps > nsubdivisions, "CFL condition must be fullfilled.");

  let mesh = Rc::new(unit_hypercube_mesh(d, nsubdivisions));
  let nnodes = mesh.nnodes();

  let space = Rc::new(FeSpace::new(mesh.clone()));

  let mut galmat_laplacian = assemble_galmat_lagrangian(&space, laplacian_neg_elmat);
  let mut galmat_mass = assemble_galmat_lagrangian(&space, lumped_mass_elmat);
  let mut galvec = assemble_galvec(&space, LoadElvec::new(|_| 0.0));

  fix_dof_coeffs(
    |idof| (idof == 0 || idof == nnodes - 1).then_some(0.0),
    &mut galmat_laplacian,
    &mut galvec,
  );
  fix_dof_coeffs(
    |idof| (idof == 0 || idof == nnodes - 1).then_some(0.0),
    &mut galmat_mass,
    &mut galvec,
  );

  let galmat_mass_cholesky = nas::factorization::CscCholesky::factor(&galmat_mass).unwrap();

  let mut file = std::fs::File::create("out/wavesol.txt").unwrap();
  std::io::Write::write_all(&mut file, format!("{} {}\n", d, nnodes).as_bytes()).unwrap();

  let mut mu = na::DVector::zeros(nnodes);
  let mut nu = na::DVector::zeros(nnodes);

  for inode in 0..nnodes {
    let x = inode as f64 / (nnodes - 1) as f64;
    mu[inode] = (x * TAU).sin();
    //nu[inode] = -TAU * (x * TAU).sin();
  }

  let contents: String = mu.iter().map(|v| format!("{v}\n")).collect();
  std::io::Write::write_all(&mut file, contents.as_bytes()).unwrap();

  for istep in 1..nsteps {
    println!("step={istep}");

    let rhs = &galmat_mass * nu + tau * (&galvec - &galmat_laplacian * &mu);
    nu = galmat_mass_cholesky.solve(&rhs).column(0).into();
    mu = mu + tau * &nu;

    let contents: String = mu.iter().map(|v| format!("{v}\n")).collect();
    std::io::Write::write_all(&mut file, contents.as_bytes()).unwrap();
  }
}
