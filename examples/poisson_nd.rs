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

  // Spatial dimension of the problem.
  let d: usize = 2;

  let kstart = 0;
  let kend = 15;
  let klen = kend - kstart + 1;
  let mut errors = Vec::with_capacity(klen);
  for k in kstart..=kend {
    let expk = 2usize.pow(k as u32);

    // Create mesh of unit hypercube $[0, 1]^d$.
    let subdivisions = expk;
    let cells_per_dim = subdivisions;
    let nodes_per_dim = subdivisions + 1;
    let ncells = cells_per_dim.pow(d as u32);
    let nnodes = nodes_per_dim.pow(d as u32);
    let mesh = unit_hypercube_mesh(d, subdivisions);
    let mesh = Rc::new(mesh);

    // Analytical solution.
    let analytical_sol = |x: na::DVectorView<f64>| (x.iter().product::<f64>()).exp();
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

      prefactor * analytical_sol(x)
    };

    // Create FE space.
    let space = Rc::new(FeSpace::new(mesh.clone()));
    let ndofs = space.ndofs();

    // Assemble galerkin matrix and galerkin vector.
    let mut galmat = assemble_galmat_lagrangian(space.clone(), laplacian_neg_elmat);
    let mut galvec = assemble_galvec(space, LoadElvec::new(|x| -analytic_laplacian(x)));

    // Enforce homogeneous Dirichlet boundary conditions
    // by fixing dofs on boundary for any dimension d.
    assemble::fix_dof_coeffs(
      |mut idof| {
        let mut fcoord = na::DVector::zeros(d);
        let mut is_boundary = false;

        for dim in 0..d {
          let icoord = idof % nodes_per_dim;
          fcoord[dim] = icoord as f64 / (nodes_per_dim as f64 - 1.0);

          if icoord == 0 || icoord == nodes_per_dim - 1 {
            is_boundary = true;
            break;
          }
          idof /= nodes_per_dim;
        }

        is_boundary.then_some(analytical_sol(fcoord.as_view()))
      },
      &mut galmat,
      &mut galvec,
    );

    // Obtain galerkin solution by solving LSE.
    let galsol = nas::factorization::CscCholesky::factor(&galmat)
      .unwrap()
      .solve(&galvec);

    // Compute analytical solution on mesh nodes.
    let analytical_sol =
      na::DVector::from_iterator(nnodes, mesh.node_coords().column_iter().map(analytical_sol));

    // Compute L2 error.
    let diff = analytical_sol - galsol;
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
    println!(
      "\
      nsubdivisions: {subdivisions:4}, \
      ncells: {ncells:6}, \
      ndofs: {ndofs:6}, \
      error: {error:9.3e}, \
      conv_rate: {conv_rate:6.2} \
      "
    );
  }
}
