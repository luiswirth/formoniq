extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use formoniq::{
  assemble::{assemble_galmat_lagrangian, drop_dofs_galmat},
  fe::{laplacian_neg_elmat, lumped_mass_elmat},
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

  drop_dofs_galmat(
    |idof| idof == 0 || idof == nnodes - 1,
    &mut galmat_laplacian,
  );
  drop_dofs_galmat(|idof| idof == 0 || idof == nnodes - 1, &mut galmat_mass);

  let ndofs = nnodes - 2;

  let galmat_mass_cholesky = nas::factorization::CscCholesky::factor(&galmat_mass).unwrap();

  let mut file = std::fs::File::create("out/wavesol.txt").unwrap();
  std::io::Write::write_all(&mut file, format!("{} {}\n", d, ndofs).as_bytes()).unwrap();

  let mut mu = na::DVector::zeros(ndofs);
  let mut nu = na::DVector::zeros(ndofs);

  for inode in 1..(1 + ndofs) {
    let idof = inode - 1;
    let x = inode as f64 / (nnodes - 1) as f64;
    mu[idof] = (x * TAU).sin();
  }

  let contents: String = mu.iter().map(|v| format!("{v}\n")).collect();
  std::io::Write::write_all(&mut file, contents.as_bytes()).unwrap();

  for istep in 1..nsteps {
    println!("step={istep}");

    nu = galmat_mass_cholesky
      .solve(&(galmat_mass.clone() * nu.clone() - tau * galmat_laplacian.clone() * mu.clone()))
      .column(0)
      .into();

    mu = mu + tau * nu.clone();

    let contents: String = mu.iter().map(|v| format!("{v}\n")).collect();
    std::io::Write::write_all(&mut file, contents.as_bytes()).unwrap();
  }
}
