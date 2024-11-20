use std::rc::Rc;

use matrix::FaerCholesky;
use mesh::SimplicialManifold;
use space::FeSpace;

extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

pub mod assemble;
pub mod combinatorics;
pub mod fe;
pub mod geometry;
pub mod matrix;
pub mod mesh;
pub mod space;
pub mod util;

pub type Dim = usize;
pub type Codim = usize;

pub type VertexIdx = usize;

pub fn solve_poisson<F>(
  mesh: &Rc<SimplicialManifold>,
  load_data: na::DVector<f64>,
  boundary_data: F,
) -> na::DVector<f64>
where
  F: Fn(VertexIdx) -> f64,
{
  let space = FeSpace::new(Rc::clone(mesh));

  let elmat = fe::laplacian_neg_elmat;
  let mut galmat = assemble::assemble_galmat(&space, elmat);

  let elvec = fe::LoadElvec::new(load_data);
  let mut galvec = assemble::assemble_galvec(&space, elvec);

  assemble::enforce_dirichlet_bc(mesh, boundary_data, &mut galmat, &mut galvec);

  let galmat = galmat.to_nalgebra_csc();
  FaerCholesky::new(galmat).solve(&galvec)
}
