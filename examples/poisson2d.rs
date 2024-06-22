extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use formoniq::{
  assemble::{self, assemble_galmat_lagrangian, assemble_galvec},
  mesh::factory,
  space::FeSpace,
};

use std::rc::Rc;

fn main() {
  tracing_subscriber::fmt::init();

  // Create mesh of unit square [0, 1]^2.
  let quads_per_dim: usize = 100;
  let nquads = quads_per_dim.pow(2);
  let nodes_per_dim = quads_per_dim + 1;
  let nnodes = nodes_per_dim.pow(2);
  let ncells = nquads * 2;
  let mut nodes = Vec::with_capacity(nnodes);
  for iy in 0..nodes_per_dim {
    let y = iy as f64 / (nodes_per_dim - 1) as f64;
    for ix in 0..nodes_per_dim {
      let x = ix as f64 / (nodes_per_dim - 1) as f64;
      nodes.push(na::DVector::from_column_slice(&[x, y]));
    }
  }
  let nodes = na::DMatrix::from_columns(&nodes);
  let mut cells = Vec::with_capacity(ncells);
  for iy in 0..quads_per_dim {
    for ix in 0..quads_per_dim {
      cells.push(vec![
        ix + iy * nodes_per_dim,
        (ix + 1) + iy * nodes_per_dim,
        (ix + 1) + (iy + 1) * nodes_per_dim,
      ]);
      cells.push(vec![
        (ix + 1) + (iy + 1) * nodes_per_dim,
        ix + (iy + 1) * nodes_per_dim,
        ix + iy * nodes_per_dim,
      ]);
    }
  }
  let mesh = factory::from_facets(nodes.clone(), cells);
  let mesh = Rc::new(mesh);

  // Create FE space.
  let space = Rc::new(FeSpace::new(mesh));
  let ndofs = space.ndofs();

  // Assemble galerkin matrix and galerkin vector.
  let mut galmat = assemble_galmat_lagrangian(space.clone());
  let mut galvec = assemble_galvec(space, |_| 1.0);

  // Enforce homogeneous dirichlet boundary conditions
  // by dropping dofs on boundary.
  assemble::fix_dof_coeffs(
    |idof| {
      if idof < nodes_per_dim
        || idof % nodes_per_dim == 0
        || (idof + 1) % nodes_per_dim == 0
        || idof >= nnodes - nodes_per_dim
      {
        Some(0.0)
      } else {
        None
      }
    },
    &mut galmat,
    &mut galvec,
  );

  // Obtain galerkin solution by solving LSE.
  let galsol = nas::factorization::CscCholesky::factor(&galmat)
    .unwrap()
    .solve(&galvec);

  // Compute exact analytical solution on nodes.
  let exact_sol = |x: na::DVectorView<f64>| 0.25 * (x[0] * x[0] - x[0]) * (x[1] * x[1] - x[1]);
  println!("{nodes}");
  let exact_sol = na::DVector::from_iterator(ndofs, nodes.column_iter().map(exact_sol));

  for (a, e) in galsol.iter().zip(exact_sol.iter()) {
    println!("a={a:.3}, e={e:.3}");
  }

  // Compute error norm.
  let error = exact_sol - galsol;
  let error = error.norm();
  println!("error = {:e}", error);
}
