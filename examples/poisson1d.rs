extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use formoniq::{
  assemble::{self, assemble_galmat_lagrangian, assemble_galvec},
  fe::{laplacian_neg_elmat, LoadElvec},
  mesh::factory,
  space::FeSpace,
};

use std::{f64::consts::E, rc::Rc};

// $Delta u = e^x <=> -Delta u = -e^x$ on $[0, 1]$
// $u(0) = u(1) = 0$ (Homogenegous Dirichlet B.C.)
fn main() {
  tracing_subscriber::fmt::init();

  let kstart = 0;
  let kend = 19;
  let klen = kend - kstart + 1;
  let mut errors = Vec::with_capacity(klen);
  for k in kstart..=kend {
    let expk = 2usize.pow(k as u32);

    // Create mesh of unit interval [0, 1].
    let ncells = expk;
    let nnodes = ncells + 1;
    let nodes: Vec<_> = (0..nnodes)
      .map(|c| c as f64 / (nnodes - 1) as f64)
      .collect();
    let nodes = na::DMatrix::from_column_slice(1, nodes.len(), &nodes);
    let cells: Vec<_> = (0..nnodes)
      .collect::<Vec<_>>()
      .windows(2)
      .map(|w| w.to_vec())
      .collect();
    assert!(cells.len() == ncells);
    let mesh = factory::from_facets(nodes.clone(), cells, false);
    let mesh = Rc::new(mesh);

    // Create FE space.
    let space = Rc::new(FeSpace::new(mesh.clone()));
    let ndofs = space.ndofs();

    // Assemble galerkin matrix and galerkin vector.
    let mut galmat = assemble_galmat_lagrangian(&space, laplacian_neg_elmat);
    let mut galvec = assemble_galvec(&space, LoadElvec::new(|x| -x[0].exp()));

    // Enforce homogeneous dirichlet boundary conditions
    // by fixing dofs on boundary.
    assemble::fix_dof_coeffs(
      |idof| (idof == 0 || idof == ndofs - 1).then_some(0.0),
      &mut galmat,
      &mut galvec,
    );

    // Obtain galerkin solution by solving LSE.
    let galsol = nas::factorization::CscCholesky::factor(&galmat)
      .unwrap()
      .solve(&galvec);

    // Compute exact analytical solution on nodes.
    let exact_sol = |x: f64| x.exp() + (1.0 - E) * x - 1.0;
    let exact_sol = nodes.map(exact_sol).transpose();

    // Compute L2 error.
    let diff = exact_sol.clone() - galsol.clone();
    let mut error = 0.0;
    for (icell, cell) in mesh.dsimplicies(1).iter().enumerate() {
      let mut sum = 0.0;
      for &ivertex in cell.vertices() {
        sum += diff[ivertex].powi(2);
      }
      let nvertices = cell.nvertices();
      let vol = mesh.coordinate_simplex((1, icell)).vol();
      let local_error = (vol / nvertices as f64) * sum;
      error += local_error;
    }
    error = error.sqrt();

    let conv_rate = if let Some(&prev_error) = errors.last() {
      let quot: f64 = error / prev_error;
      -quot.log2()
    } else {
      f64::NAN
    };

    errors.push(error);
    println!("ncells: {ncells:6} error: {error:9.3e} conv_rate: {conv_rate:6.2}");
  }
}
