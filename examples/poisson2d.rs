extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use formoniq::{
  assemble::{self, assemble_galmat_lagrangian, assemble_galvec},
  fe::{laplacian_neg_elmat, LoadElvec},
  mesh::factory::unit_hypercube_mesh,
  space::FeSpace,
};

use std::rc::Rc;

fn main() {
  tracing_subscriber::fmt::init();

  let d = 2usize;

  let kstart = 0;
  let kend = 8;
  let klen = kend - kstart + 1;
  let mut errors = Vec::with_capacity(klen);
  for k in kstart..=kend {
    let expk = 2usize.pow(k as u32);

    // Create mesh of unit square [0, 1]^2.
    let quads_per_dim = expk;
    let nquads = quads_per_dim.pow(d as u32);
    let nodes_per_dim = quads_per_dim + 1;
    let nnodes = nodes_per_dim.pow(d as u32);
    let ncells = nquads * 2;
    let mesh = unit_hypercube_mesh(d, quads_per_dim);
    let mesh = Rc::new(mesh);

    // Create FE space.
    let space = Rc::new(FeSpace::new(mesh.clone()));
    let ndofs = space.ndofs();

    // Assemble galerkin matrix and galerkin vector.
    let mut galmat = assemble_galmat_lagrangian(space.clone(), laplacian_neg_elmat);
    let mut galvec = assemble_galvec(
      space,
      LoadElvec::new(|x| -x.norm_squared() * (x[0] * x[1]).exp()),
    );

    // Enforce homogeneous dirichlet boundary conditions
    // by fixing dofs on boundary.
    assemble::fix_dof_coeffs(
      |idof| {
        if idof < nodes_per_dim {
          // bottom
          Some(1.)
        } else if idof % nodes_per_dim == 0 {
          // left
          Some(1.)
        } else if (idof + 1) % nodes_per_dim == 0 {
          // right
          let y = ((idof + 1) / nodes_per_dim - 1) as f64 / nodes_per_dim as f64;
          Some(y.exp())
        } else if idof >= nnodes - nodes_per_dim {
          // top
          let x = (idof - (nnodes - nodes_per_dim)) as f64 / nodes_per_dim as f64;
          Some(x.exp())
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
    let exact_sol = |x: na::DVectorView<f64>| (x[0] * x[1]).exp();
    let exact_sol =
      na::DVector::from_iterator(ndofs, mesh.node_coords().column_iter().map(exact_sol));

    // Compute L2 error.
    let diff = exact_sol - galsol;
    let mut error = 0.0;
    for (icell, cell) in mesh.dsimplicies(d).iter().enumerate() {
      let mut sum = 0.0;
      for &ivertex in cell.vertices() {
        sum += diff[ivertex].powi(2);
      }
      let nvertices = cell.nvertices();
      let vol = mesh.coordinate_simplex((d, icell)).vol();
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
