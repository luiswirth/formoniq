//! Solves a manufactored poisson problem in d dimensions
//! and determines the algebraic convergence rate.

extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use formoniq::{
  assemble::{self, assemble_galmat, assemble_galvec},
  fe::{l2_norm, LoadElvec, UpwindAdvectionElmat},
  matrix::FaerLu,
  mesh::hyperbox::HyperBoxMesh,
  space::FeSpace,
};

use std::{f64::consts::PI, fmt::Write, rc::Rc};

fn main() {
  tracing_subscriber::fmt::init();

  let dim: usize = 2;
  println!("Stationary Linear Advection in {dim}D");

  let dirichlet_data = |x: na::DVectorView<f64>| {
    if x[1] == 0.0 {
      (PI * x[0]).sin()
    } else {
      0.0
    }
  };

  let velocity_field = |x: na::DVectorView<f64>| na::DVector::from_column_slice(&[-x[1], x[0]]);

  let anal_sol = |x: na::DVectorView<f64>| {
    if x.norm() < 1.0 {
      dirichlet_data((&na::Vector2::new(x.norm(), 0.0)).into())
    } else {
      0.0
    }
  };

  let kstart = 0;
  let kend = 10;
  let klen = kend - kstart + 1;

  fn print_seperator() {
    let nchar = 78;
    println!("{}", "-".repeat(nchar));
  }

  print_seperator();
  println!(
    "| {:>2} | {:>13} | {:>10} | {:>16} | {:>9} | {:>9} |",
    "k", "nsubdivisions", "mesh width", "shape regularity", "L2 error", "conv rate"
  );
  print_seperator();

  let mut errors = Vec::with_capacity(klen);
  for k in kstart..=kend {
    let expk = 2usize.pow(k as u32);
    let nboxes_per_dim = expk;

    // Create mesh of unit hypercube $[0, 1]^d$.
    let mesh = HyperBoxMesh::new_unit(dim, nboxes_per_dim);
    let mesh_width = mesh.mesh().mesh_width();
    let shape_regularity = mesh.mesh().shape_regularity_measure();

    // Create FE space.
    let space = Rc::new(FeSpace::new(mesh.mesh().clone()));

    let mut galmat = assemble_galmat(
      &space,
      UpwindAdvectionElmat::new(velocity_field, space.mesh()),
    );
    let mut galvec = assemble_galvec(&space, LoadElvec::new(|_| 0.0));

    let dirichlet_map = |idof| {
      let inode = idof;
      let icoords = mesh.info().node_cart_idx(inode);
      let fcoords = mesh.info().node_pos(inode);
      let is_boundary = mesh.is_node_on_boundary(inode);
      let is_inflow = icoords[1] == 0 || icoords[0] == mesh.info().nnodes_per_dim() - 1;
      let is_imposable = is_boundary && is_inflow;
      is_imposable.then_some(dirichlet_data(fcoords.as_view()))
    };

    assemble::fix_dof_coeffs(dirichlet_map, &mut galmat, &mut galvec);

    let galmat = galmat.to_nalgebra();

    let galmat_lu = FaerLu::new(galmat);
    let galsol = galmat_lu.solve(&galvec);

    if k == kend {
      let mut file = std::fs::File::create("out/advection_stationary_sol.txt").unwrap();
      std::io::Write::write_all(
        &mut file,
        format!("{} {}\n", dim, nboxes_per_dim + 1).as_bytes(),
      )
      .unwrap();
      let contents: String = galsol.row_iter().fold(String::new(), |mut s, v| {
        let _ = writeln!(s, "{}", v[0]);
        s
      });
      std::io::Write::write_all(&mut file, contents.as_bytes()).unwrap();
    }

    let anal_sol_mesh = na::DVector::from_iterator(
      space.ndofs(),
      mesh.nodes().coords().column_iter().map(anal_sol),
    );
    let diff = &galsol - &anal_sol_mesh;
    let error = l2_norm(diff, mesh.mesh());

    let conv_rate = if let Some(&prev_error) = errors.last() {
      let quot: f64 = error / prev_error;
      -quot.log2()
    } else {
      f64::INFINITY
    };
    errors.push(error);

    println!(
      "| {:>2} | {:>13} | {:>10.3e} | {:>16.3e} | {:>9.3e} | {:>9.2} |",
      k, nboxes_per_dim, mesh_width, shape_regularity, error, conv_rate
    );
  }
  print_seperator();
}
