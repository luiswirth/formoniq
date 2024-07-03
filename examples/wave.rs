extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use formoniq::{
  assemble::{assemble_galmat_lagrangian, drop_dofs_galmat},
  fe::{laplacian_neg_elmat, lumped_mass_elmat},
  mesh::factory::unit_hypercube_mesh,
  space::FeSpace,
};
use nas::CscMatrix;

use std::{f64::consts::PI, rc::Rc};

fn main() {
  tracing_subscriber::fmt::init();

  let d: usize = 1;
  let nsubdivisions = 100;
  let tau = 0.0001;
  let nsteps = 1000;

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

  let mut lse_a = nas::CooMatrix::new(ndofs * 2, ndofs * 2);
  for i in 0..ndofs {
    lse_a.push(i, i, 1.0);
    lse_a.push(i, ndofs + i, -0.5 * tau);
  }
  for (r, c, &v) in galmat_laplacian.triplet_iter() {
    lse_a.push(ndofs + r, c, 0.5 * tau * v);
  }
  for (r, c, &v) in galmat_mass.triplet_iter() {
    lse_a.push(ndofs + r, ndofs + c, v);
  }
  let lse_a = CscMatrix::from(&lse_a);
  let lse_a_cholesky = nas::factorization::CscCholesky::factor(&lse_a).unwrap();

  let mut lse_b = nas::CooMatrix::new(ndofs * 2, ndofs * 2);
  for i in 0..ndofs {
    lse_b.push(i, i, 1.0);
    lse_b.push(i, ndofs + i, 0.5 * tau);
  }
  for (r, c, &v) in galmat_laplacian.triplet_iter() {
    lse_b.push(ndofs + r, c, -0.5 * tau * v);
  }
  for (r, c, &v) in galmat_mass.triplet_iter() {
    lse_b.push(ndofs + r, ndofs + c, v);
  }
  let lse_b = CscMatrix::from(&lse_b);

  let mut munu = na::DVector::zeros(2 * ndofs);
  for inode in 1..(1 + ndofs) {
    let idof = inode - 1;
    let x = inode as f64 / (nnodes - 1) as f64;
    munu[idof] = (x * PI).sin();
  }

  let mut file = std::fs::File::create("out/wavesol.txt").unwrap();
  std::io::Write::write_all(&mut file, format!("{} {}\n", d, ndofs).as_bytes()).unwrap();

  for istep in 0..nsteps {
    println!("step={istep}");
    munu = lse_a_cholesky
      .solve(&(lse_b.clone() * munu))
      .column(0)
      .into();

    let mu = munu.view_range(0..ndofs, ..);

    let contents: String = mu.iter().map(|v| format!("{v}\n")).collect();
    std::io::Write::write_all(&mut file, contents.as_bytes()).unwrap();
  }
}
