//! Solves a manufactored poisson problem in d dimensions
//! and determines the algebraic convergence rate.

extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use formoniq::{
  assemble::{self, assemble_galmat, assemble_galvec},
  fe::{self, LoadElvec, UpwindAdvectionElmat},
  matrix::FaerLu,
  mesh::hypercube::{
    hypercube_mesh, linear_idx2cartesian_coords, linear_idx2cartesian_idx, Hypercube,
  },
  space::FeSpace,
};

use std::rc::Rc;

fn main() {
  tracing_subscriber::fmt::init();

  let d: usize = 2;
  let nsubdivisions = 500;
  let nodes_per_dim = nsubdivisions + 1;

  let final_time = 30.0;
  let nsteps = 1000;
  let timestep = final_time / nsteps as f64;

  let dirichlet_data = |_: na::DVectorView<f64>| 0.0;

  let velocity_field = |x: na::DVectorView<f64>| na::DVector::from_column_slice(&[-x[1], x[0]]);

  let cube = Hypercube::new_min_max(
    na::DVector::from_element(d, -1.0),
    na::DVector::from_element(d, 1.0),
  );
  let mesh = hypercube_mesh(&cube, nsubdivisions);
  let mesh = Rc::new(mesh);

  let space = Rc::new(FeSpace::new(mesh.clone()));
  let ndofs = space.ndofs();

  let dirichlet_map = |idof| {
    let cart_idx = linear_idx2cartesian_idx(idof, d, nodes_per_dim);
    let card_coords = linear_idx2cartesian_coords(idof, &cube, nodes_per_dim);
    let is_boundary = cart_idx.iter().any(|&c| c == 0 || c == nodes_per_dim - 1);
    let is_inflow = false;
    let is_imposable = is_boundary && is_inflow;
    is_imposable.then_some(dirichlet_data(card_coords.as_view()))
  };

  let mut galvec = assemble_galvec(&space, LoadElvec::new(|_| 0.0));

  let mut galmat_mass = assemble_galmat(&space, fe::lumped_mass_elmat);
  assemble::fix_dof_coeffs(dirichlet_map, &mut galmat_mass, &mut galvec);
  let galmat_mass = galmat_mass.to_nalgebra();

  let mut galmat_advection = assemble_galmat(
    &space,
    UpwindAdvectionElmat::new(velocity_field, space.mesh()),
  );
  assemble::fix_dof_coeffs(dirichlet_map, &mut galmat_advection, &mut galvec);
  let galmat_advection = galmat_advection.to_nalgebra();

  let galmat = &galmat_mass + timestep * &galmat_advection;

  let galmat_lu = FaerLu::new(galmat);

  let mut mu = na::DVector::zeros(ndofs);
  for idof in 0..ndofs {
    let x = linear_idx2cartesian_coords(idof, &cube, nodes_per_dim);
    let alpha = 10.0;
    let offset = na::DVector::from_column_slice(&[0.5, 0.0]);
    let norm = (x - offset).norm_squared();
    mu[idof] = (-alpha * norm).exp()
  }

  let mut file = std::fs::File::create("out/advection_transient_sol.txt").unwrap();
  std::io::Write::write_all(
    &mut file,
    format!("{} {} {} {}\n", d, nodes_per_dim, final_time, nsteps).as_bytes(),
  )
  .unwrap();
  let contents: String = mu.row_iter().map(|v| format!("{}\n", v[0])).collect();
  std::io::Write::write_all(&mut file, contents.as_bytes()).unwrap();

  for istep in 1..nsteps {
    println!("step={istep}");

    let rhs = &galmat_mass * mu + timestep * &galvec;
    mu = galmat_lu.solve(&rhs).column(0).into();

    let contents: String = mu.row_iter().map(|v| format!("{}\n", v[0])).collect();
    std::io::Write::write_all(&mut file, contents.as_bytes()).unwrap();
  }
}
