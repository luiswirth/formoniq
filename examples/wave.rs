extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use formoniq::{
  assemble::{assemble_galmat_lagrangian, assemble_galvec, fix_dof_coeffs},
  fe::{laplacian_neg_elmat, lumped_mass_elmat, LoadElvec},
  mesh::factory::unit_hypercube_mesh,
  space::FeSpace,
};

use std::rc::Rc;

fn main() {
  tracing_subscriber::fmt::init();

  let d: usize = 1;
  let nsubdivisions = 100;

  let tfinal = 2.0;
  let nsteps = 200;
  let tau = tfinal / nsteps as f64;

  assert!(nsteps > nsubdivisions, "CFL condition must be fullfilled.");

  let mesh = Rc::new(unit_hypercube_mesh(d, nsubdivisions));
  let nnodes = mesh.nnodes();

  let space = Rc::new(FeSpace::new(mesh.clone()));

  let galmat_laplacian = assemble_galmat_lagrangian(&space, laplacian_neg_elmat);
  let galmat_mass = assemble_galmat_lagrangian(&space, lumped_mass_elmat);
  let galvec = assemble_galvec(&space, LoadElvec::new(|_| 0.0));

  let mut file = std::fs::File::create("out/wavesol.txt").unwrap();
  std::io::Write::write_all(&mut file, format!("{} {}\n", d, nnodes).as_bytes()).unwrap();

  let mut mu = na::DVector::zeros(nnodes);
  let mut nu = na::DVector::zeros(nnodes);

  for inode in 0..nnodes {
    mu[inode] = 0.0;
  }

  let contents: String = mu.iter().map(|v| format!("{v}\n")).collect();
  std::io::Write::write_all(&mut file, contents.as_bytes()).unwrap();

  for istep in 1..nsteps {
    let t = istep as f64 / (nsteps - 1) as f64 * tfinal;
    println!("step={istep}");

    let bc = |idof| {
      if idof == 0 {
        Some(0.0)
      } else if idof == nnodes - 1 {
        let x = if t < 0.2 {
          t * (0.2 - t) * 10. * (10.0 * t).sin()
        } else if t > 1.0 && t < 1.2 {
          (t - 1.0) * (1.2 - t) * 10. * (10.0 * (t - 1.0)).sin()
        } else {
          0.
        };
        dbg!(x);
        Some(x)
      } else {
        None
      }
    };

    let mut galmat_laplacian = galmat_laplacian.clone();
    let mut galmat_mass = galmat_mass.clone();
    let mut galvec = galvec.clone();

    fix_dof_coeffs(bc, &mut galmat_laplacian, &mut galvec);
    fix_dof_coeffs(bc, &mut galmat_mass, &mut galvec);

    let galmat_mass_cholesky = nas::factorization::CscCholesky::factor(&galmat_mass).unwrap();

    let rhs = &galmat_mass * nu + tau * (&galvec - &galmat_laplacian * &mu);
    nu = galmat_mass_cholesky.solve(&rhs).column(0).into();
    mu = mu + tau * &nu;

    let contents: String = mu.iter().map(|v| format!("{v}\n")).collect();
    std::io::Write::write_all(&mut file, contents.as_bytes()).unwrap();
  }
}
