//! Solves a manufactored poisson problem in d dimensions
//! and determines the algebraic convergence rate.

extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use formoniq::{
  assemble::{self, assemble_galmat, assemble_galvec},
  fe::{l2_norm, laplacian_neg_elmat, LoadElvec},
  matrix::FaerCholesky,
  mesh::hypercube::{hypercube_mesh, HyperRectangle},
  space::FeSpace,
};

use std::fmt::Write;
use std::rc::Rc;

fn main() {
  tracing_subscriber::fmt::init();

  // Spatial dimension of the problem.
  let d: usize = 2;

  let kstart = 0;
  let kend = 10;
  let klen = kend - kstart + 1;

  // Define analytic solution.
  // $u = exp(x_1 x_2 dots x_n)$
  let analytic_sol = |x: na::DVectorView<f64>| (x.iter().product::<f64>()).exp();
  let analytic_laplacian = |x: na::DVectorView<f64>| {
    let mut prefactor = 0.0;

    for i in 0..d {
      let mut partial_product = 1.0;
      for j in 0..d {
        if i != j {
          partial_product *= x[j].powi(2);
        }
      }
      prefactor += partial_product;
    }

    prefactor * analytic_sol(x)
  };

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
    let nsubdivisions = expk;

    // Create mesh of unit hypercube $[0, 1]^d$.
    let cube = HyperRectangle::new_unit(d);
    let mesh = hypercube_mesh(&cube, nsubdivisions);
    let mesh_width = mesh.mesh_width();
    let shape_regularity = mesh.shape_regularity_measure();

    // Create FE space.
    let space = Rc::new(FeSpace::new(mesh.clone()));

    // Compute Galerkin solution to manufactored poisson problem.
    let galsol = solve_poisson(space, analytic_sol, analytic_laplacian);

    if k == kend {
      let mut file = std::fs::File::create("out/galsol.txt").unwrap();
      let contents: String = galsol.row_iter().fold(String::new(), |mut s, v| {
        let _ = writeln!(s, "{}", v[0]);
        s
      });
      std::io::Write::write_all(
        &mut file,
        format!("{} {}\n", d, nsubdivisions + 1).as_bytes(),
      )
      .unwrap();
      std::io::Write::write_all(&mut file, contents.as_bytes()).unwrap();
    }

    // Compute analytical solution on mesh nodes.
    let analytical_sol = na::DVector::from_iterator(
      mesh.nnodes(),
      mesh.node_coords().column_iter().map(analytic_sol),
    );

    // Compute L2 error and convergence rate.
    let error = l2_norm(analytical_sol - galsol, &mesh);
    let conv_rate = if let Some(&prev_error) = errors.last() {
      let quot: f64 = error / prev_error;
      -quot.log2()
    } else {
      f64::INFINITY
    };
    errors.push(error);

    println!(
      "| {:>2} | {:>13} | {:>10.3e} | {:>16.3e} | {:>9.3e} | {:>9.2} |",
      k, nsubdivisions, mesh_width, shape_regularity, error, conv_rate
    );
  }
  print_seperator();
}

fn solve_poisson<F, G>(
  space: Rc<FeSpace>,
  analytic_sol: F,
  analytic_laplacian: G,
) -> na::DVector<f64>
where
  F: Fn(na::DVectorView<f64>) -> f64,
  G: Fn(na::DVectorView<f64>) -> f64,
{
  let mesh = space.mesh().clone();
  let d = mesh.dim_ambient();

  // Assemble galerkin matrix and galerkin vector.
  let mut galmat = assemble_galmat(&space, laplacian_neg_elmat);
  let mut galvec = assemble_galvec(&space, LoadElvec::new(|x| -analytic_laplacian(x)));

  // Enforce homogeneous Dirichlet boundary conditions
  // by fixing dofs on boundary.
  let nodes_per_dim = (mesh.nnodes() as f64).powf((d as f64).recip()) as usize;
  assemble::fix_dof_coeffs(
    |mut idof| {
      let mut fcoord = na::DVector::zeros(d);
      let mut is_boundary = false;
      for dim in 0..d {
        let icoord = idof % nodes_per_dim;
        fcoord[dim] = icoord as f64 / (nodes_per_dim - 1) as f64;
        is_boundary |= icoord == 0 || icoord == nodes_per_dim - 1;
        idof /= nodes_per_dim;
      }

      is_boundary.then_some(analytic_sol(fcoord.as_view()))
    },
    &mut galmat,
    &mut galvec,
  );

  let galmat = galmat.to_nalgebra();

  // Obtain Galerkin solution by solving LSE.
  let galsol = FaerCholesky::new(galmat).solve(&galvec).column(0).into();

  galsol
}
