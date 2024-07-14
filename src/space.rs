use crate::mesh::{EntityId, Mesh};

use std::rc::Rc;

pub type DofId = usize;

pub struct FeSpace {
  mesh: Rc<Mesh>,
}

impl FeSpace {
  pub fn new(mesh: Rc<Mesh>) -> Self {
    Self { mesh }
  }

  pub fn mesh(&self) -> &Rc<Mesh> {
    &self.mesh
  }

  /// The number of degrees of freedoms _associated_ with simplicies of the given dimension.
  pub fn ndofs(&self) -> usize {
    self.mesh.dsimplicies(0).len()
  }

  /// Lagrangian dofs
  pub fn dof_indices_global(&self, simplex: EntityId) -> Vec<DofId> {
    let vertices = self.mesh.simplex_by_id(simplex).vertices();
    vertices.to_vec()
  }

  pub fn dof_pos(&self, idof: DofId) -> na::DVector<f64> {
    self.mesh.node_coords().column(idof).into()
  }
}
