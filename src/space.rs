use crate::{
  mesh::{CellIdx, SimplicialManifold},
  Dim,
};

use std::rc::Rc;

pub type DofId = usize;

/// A Finite Element Space of piecewiese-linear differential forms.
///
/// Uses the Whitney basis.
pub struct FeSpace {
  /// The rank of the differential form.
  rank: Dim,
  /// The underlying mesh of the space.
  mesh: Rc<SimplicialManifold>,
  /// Degrees-of-Freedom handler
  dof_handler: DofHandler,
}

pub struct DofHandler {
  local2global_idx: Vec<Vec<DofId>>,
}
impl DofHandler {
  pub fn new(rank: Dim, mesh: &SimplicialManifold) -> Self {
    let local2global_idx = mesh
      .cells()
      .iter()
      .map(|c| c.descendants(rank).map(|d| d.kidx()).collect())
      .collect();
    Self { local2global_idx }
  }

  pub fn local2global(&self, cell: CellIdx) -> &[DofId] {
    &self.local2global_idx[cell]
  }
}

impl FeSpace {
  pub fn new(mesh: Rc<SimplicialManifold>) -> Self {
    let rank = 0;
    let dof_handler = DofHandler::new(rank, &mesh);
    Self {
      mesh,
      rank,
      dof_handler,
    }
  }

  pub fn mesh(&self) -> &Rc<SimplicialManifold> {
    &self.mesh
  }

  pub fn rank(&self) -> Dim {
    self.rank
  }

  pub fn ndofs(&self) -> usize {
    self.mesh.skeleton(self.rank).len()
  }

  pub fn dof_handler(&self) -> &DofHandler {
    &self.dof_handler
  }
}
