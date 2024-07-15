extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use formoniq::{
  assemble::{assemble_galmat, assemble_galvec, fix_dof_coeffs},
  fe::{laplacian_neg_elmat, lumped_mass_elmat, LoadElvec},
  mesh::hypercube::{hypercube_mesh, HyperRectangle},
  space::FeSpace,
};

use std::{f64::consts::PI, rc::Rc};

fn main() {
  tracing_subscriber::fmt::init();

  let d: usize = 1;
  let nsubdivisions = 100;

  let alpha = 1.0;
  let tfinal = 1.0;
  let nsteps = 200;
  let tau = tfinal / nsteps as f64;

  let cube = HyperRectangle::new_unit(d);
  let mesh = hypercube_mesh(&cube, nsubdivisions);
  let nnodes = mesh.nnodes();

  let space = Rc::new(FeSpace::new(mesh.clone()));

  let mut galmat_laplacian = assemble_galmat(&space, laplacian_neg_elmat);
  let mut galmat_mass = assemble_galmat(&space, lumped_mass_elmat);
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

  let galmat_laplacian = galmat_laplacian.to_nalgebra();
  let galmat_mass = galmat_mass.to_nalgebra();

  let galmat = &galmat_mass + alpha * tau * &galmat_laplacian;
  let galmat_cholesky = nas::factorization::CscCholesky::factor(&galmat).unwrap();

  let mut file = std::fs::File::create("out/heatsol.txt").unwrap();
  std::io::Write::write_all(&mut file, format!("{} {}\n", d, nnodes).as_bytes()).unwrap();

  let mut mu = na::DVector::zeros(nnodes);

  for inode in 0..nnodes {
    let x = inode as f64 / (nnodes - 1) as f64;
    mu[inode] = (x * PI).sin();
  }

  let contents: String = mu.iter().map(|v| format!("{v}\n")).collect();
  std::io::Write::write_all(&mut file, contents.as_bytes()).unwrap();

  for istep in 1..nsteps {
    println!("step={istep}");

    let rhs = &galmat_mass * mu + tau * &galvec;
    mu = galmat_cholesky.solve(&rhs).column(0).into();

    let contents: String = mu.iter().map(|v| format!("{v}\n")).collect();
    std::io::Write::write_all(&mut file, contents.as_bytes()).unwrap();
  }
}
