//! Solves a manufactored poisson problem in d dimensions
//! and determines the algebraic convergence rate.

extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use formoniq::{
  assemble::{self, assemble_galmat, assemble_galvec},
  fe::{self, LoadElvec, UpwindAdvectionElmat},
  matrix::FaerLu,
  mesh::hyperbox::HyperBoxMesh,
  space::FeSpace,
};

use std::{f64::consts::PI, fmt::Write, rc::Rc};

fn main() {
  tracing_subscriber::fmt::init();

  let d: usize = 2;
  let nboxes_per_dim = 50;
  let nodes_per_dim = nboxes_per_dim + 1;

  let final_time = 5.0;
  let nsteps = 150;
  let timestep = final_time / nsteps as f64;

  let dirichlet_data = |_: na::DVectorView<f64>| 0.0;

  let velocity_field = |x: na::DVectorView<f64>| {
    na::DVector::from_column_slice(&[
      -(PI * x[0]).sin() * (PI * x[1]).cos(),
      (PI * x[1]).sin() * (PI * x[0]).cos(),
    ])
  };

  let mesh = HyperBoxMesh::new_unit(d, nboxes_per_dim);

  let space = Rc::new(FeSpace::new(mesh.mesh().clone()));
  let ndofs = space.ndofs();

  let dirichlet_map = |idof| {
    let inode = idof;
    let is_boundary = mesh.is_node_on_boundary(inode);
    let is_inflow = false;
    let is_imposable = is_boundary && is_inflow;
    let pos = mesh.nodes().coord(inode);
    is_imposable.then_some(dirichlet_data(pos))
  };

  println!("assembling galvec...");
  let mut galvec = assemble_galvec(&space, LoadElvec::new(|_| 0.0));

  println!("assembling galmat...");
  let mut galmat_mass = assemble_galmat(&space, fe::lumped_mass_elmat);
  assemble::fix_dof_coeffs(dirichlet_map, &mut galmat_mass, &mut galvec);
  let galmat_mass = galmat_mass.to_nalgebra();

  let mut galmat_advection = assemble_galmat(
    &space,
    UpwindAdvectionElmat::new(velocity_field, space.mesh()),
  );
  println!("fixing dofs...");
  assemble::fix_dof_coeffs(dirichlet_map, &mut galmat_advection, &mut galvec);
  let galmat_advection = galmat_advection.to_nalgebra();

  let galmat = &galmat_mass + timestep * &galmat_advection;

  println!("computing lu...");
  let galmat_lu = FaerLu::new(galmat);

  let mut mu = na::DVector::zeros(ndofs);
  for idof in 0..ndofs {
    let inode = idof;
    let x = mesh.nodes().coord(inode);
    let v = (x - na::Vector2::new(0.5, 0.25)).norm();
    mu[idof] = (1.0 - 4.0 * v).max(0.0);
  }

  let mut file = std::fs::File::create("out/advection_transient_sol.txt").unwrap();
  std::io::Write::write_all(
    &mut file,
    format!("{} {} {} {}\n", d, nodes_per_dim, final_time, nsteps).as_bytes(),
  )
  .unwrap();
  let contents: String = mu.row_iter().fold(String::new(), |mut s, v| {
    let _ = writeln!(s, "{}", v[0]);
    s
  });
  std::io::Write::write_all(&mut file, contents.as_bytes()).unwrap();

  for istep in 1..nsteps {
    println!("step={istep}");

    let rhs = &galmat_mass * mu + timestep * &galvec;
    mu = galmat_lu.solve(&rhs).column(0).into();

    let contents: String = mu.row_iter().fold(String::new(), |mut s, v| {
      let _ = writeln!(s, "{}", v[0]);
      s
    });
    std::io::Write::write_all(&mut file, contents.as_bytes()).unwrap();
  }
  println!("done!");
}
