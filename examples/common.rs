extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use formoniq::{
  assemble::{self, assemble_galmat, assemble_galvec},
  fe::{laplacian_neg_elmat, LoadElvec},
  matrix::FaerCholesky,
  mesh::{coordinates::NodeCoords, util::NodeData},
  space::FeSpace,
};

pub fn solve_manufactured_poisson<F, G>(
  nodes: &NodeCoords,
  space: &FeSpace,
  analytic_sol: F,
  analytic_laplacian: G,
) -> na::DVector<f64>
where
  F: Fn(na::DVectorView<f64>) -> f64,
  G: Fn(na::DVectorView<f64>) -> f64,
{
  let d = space.mesh().dim();

  // Assemble galerkin matrix and galerkin vector.
  let mut galmat = assemble_galmat(space, laplacian_neg_elmat);
  let mut galvec = assemble_galvec(
    space,
    LoadElvec::new(NodeData::from_coords_map(nodes, |x| -analytic_laplacian(x))),
  );

  // Enforce homogeneous Dirichlet boundary conditions
  // by fixing dofs on boundary.
  let nodes_per_dim = (nodes.nnodes() as f64).powf((d as f64).recip()) as usize;
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

  let galmat = galmat.to_nalgebra_csc();

  // Obtain Galerkin solution by solving LSE.
  let galsol = FaerCholesky::new(galmat).solve(&galvec).column(0).into();

  galsol
}
