use crate::{mesh::SimplicialManifold, Dim};

use std::rc::Rc;

pub type DofIdx = usize;

/// The Whitney Form Finite Element Space.
pub struct FeSpace {
  /// The rank of the Whitney form.
  rank: Dim,
  /// The underlying mesh of the space.
  mesh: Rc<SimplicialManifold>,
}

impl FeSpace {
  pub fn new(mesh: Rc<SimplicialManifold>) -> Self {
    let rank = 0;
    Self { mesh, rank }
  }

  pub fn rank(&self) -> Dim {
    self.rank
  }
  pub fn mesh(&self) -> &Rc<SimplicialManifold> {
    &self.mesh
  }

  pub fn ndofs(&self) -> usize {
    self.mesh.skeleton(self.rank).len()
  }
}
