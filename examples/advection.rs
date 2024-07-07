//! Solves a manufactored poisson problem in d dimensions
//! and determines the algebraic convergence rate.

extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use formoniq::{
  assemble::{self, assemble_galmat, assemble_galvec},
  fe::{LoadElvec, UpwindAdvectionElmat},
  mesh::factory::hypercube_mesh,
  space::FeSpace,
};

use std::{f64::consts::PI, rc::Rc};

fn main() {
  tracing_subscriber::fmt::init();

  let d: usize = 2;
  let nsubdivisions = 50;

  let dirichlet_data = |x: na::DVectorView<f64>| {
    if x[1] == 0.0 {
      (PI * x[0]).sin()
    } else {
      0.0
    }
  };

  let velocity_field = |x: na::DVectorView<f64>| na::DVector::from_column_slice(&[-x[1], x[0]]);

  let mesh = hypercube_mesh(d, nsubdivisions, 1.0);
  let mesh = Rc::new(mesh);

  let space = Rc::new(FeSpace::new(mesh.clone()));

  let mut galmat = assemble_galmat(
    &space,
    UpwindAdvectionElmat::new(velocity_field, space.mesh()),
  );
  let mut galvec = assemble_galvec(&space, LoadElvec::new(|_| 0.0));

  let nodes_per_dim = (mesh.nnodes() as f64).powf((d as f64).recip()) as usize;
  assemble::fix_dof_coeffs(
    |mut idof| {
      let mut icoords = na::DVector::zeros(d);
      let mut fcoords = na::DVector::zeros(d);
      let mut is_boundary = false;
      for dim in 0..d {
        let icoord = idof % nodes_per_dim;
        icoords[dim] = icoord;
        fcoords[dim] = icoord as f64 / (nodes_per_dim - 1) as f64;
        idof /= nodes_per_dim;
        is_boundary |= icoord == 0 || icoord == nodes_per_dim - 1;
      }
      let is_inflow = icoords[1] == 0 || icoords[0] == nodes_per_dim - 1;
      let is_imposable = is_boundary && is_inflow;
      is_imposable.then_some(dirichlet_data(fcoords.as_view()))
    },
    &mut galmat,
    &mut galvec,
  );

  // Obtain Galerkin solution by solving LSE.
  // TODO: Dense matrix construted!!! We need a sparse solver for non s.p.d. matrices.
  let galmat: na::DMatrix<f64> = (&galmat).into();
  let galsol: na::DVector<f64> = galmat.lu().solve(&galvec).unwrap().column(0).into();

  let mut file = std::fs::File::create("out/advectionsol.txt").unwrap();
  std::io::Write::write_all(
    &mut file,
    format!("{} {}\n", d, nsubdivisions + 1).as_bytes(),
  )
  .unwrap();
  let contents: String = galsol.iter().map(|v| format!("{v}\n")).collect();
  std::io::Write::write_all(&mut file, contents.as_bytes()).unwrap();
}
