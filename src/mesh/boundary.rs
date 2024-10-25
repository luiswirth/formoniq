use super::{CellIdx, SimplexHandle, SimplicialManifold, VertexIdx};

use itertools::Itertools;

impl SimplicialManifold {
  /// For a d-mesh computes the boundary, which consists of (d-1)-faces.
  ///
  /// The boundary simplicies are characterized by the fact that they
  /// only have 1 super entity.
  pub fn boundary_faces(&self) -> Vec<SimplexHandle> {
    self
      .faces()
      .iter()
      .filter(|f| f.antiboundary().len() == 1)
      .collect()
  }

  /// The nodes that lie on the boundary of the mesh.
  /// No particular order of nodes.
  pub fn boundary_nodes(&self) -> Vec<VertexIdx> {
    self
      .boundary_faces()
      .into_iter()
      .flat_map(|face| face.vertices().iter().copied())
      .unique()
      .collect()
  }

  pub fn boundary_cells(&self) -> Vec<CellIdx> {
    self
      .boundary_faces()
      .into_iter()
      // the boundary has only one super by definition
      .map(|face| face.antiboundary().iter().next().unwrap())
      .map(|cell| cell.kidx())
      .unique()
      .collect()
  }
}
