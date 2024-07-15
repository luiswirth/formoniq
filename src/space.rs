use crate::mesh::{CellId, SimplicialMesh};

use std::rc::Rc;

pub type DofId = usize;

pub struct FeSpace {
  mesh: Rc<SimplicialMesh>,
}

impl FeSpace {
  pub fn new(mesh: Rc<SimplicialMesh>) -> Self {
    Self { mesh }
  }

  pub fn mesh(&self) -> &Rc<SimplicialMesh> {
    &self.mesh
  }

  pub fn ndofs(&self) -> usize {
    self.mesh.nnodes()
  }

  pub fn dof_indices_global(&self, icell: CellId) -> Vec<DofId> {
    let vertices = self.mesh.cell(icell).vertices();
    vertices.to_vec()
  }

  pub fn dof_pos(&self, idof: DofId) -> na::DVector<f64> {
    self.mesh.node_coords().column(idof).into()
  }
}
