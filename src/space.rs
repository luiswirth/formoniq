use crate::mesh::{CellId, SimplicialManifold};

use std::rc::Rc;

pub type DofId = usize;

/// A Linear Lagrangian Finite Element Space
pub struct FeSpace {
  mesh: Rc<SimplicialManifold>,
}

impl FeSpace {
  pub fn new(mesh: Rc<SimplicialManifold>) -> Self {
    Self { mesh }
  }

  pub fn mesh(&self) -> &Rc<SimplicialManifold> {
    &self.mesh
  }

  pub fn ndofs(&self) -> usize {
    self.mesh.nnodes()
  }

  pub fn dof_indices_global(&self, icell: CellId) -> Vec<DofId> {
    let vertices = self.mesh.cell(icell).vertices();
    vertices.to_vec()
  }
}
