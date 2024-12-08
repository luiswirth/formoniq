use crate::util;

use super::{CellIdx, SimplexHandle, SimplicialManifold, VertexIdx};

use itertools::Itertools;

impl SimplicialManifold {
  pub fn has_boundary(&self) -> bool {
    !self.boundary_faces().is_empty()
  }

  /// For a d-mesh computes the boundary, which consists of faces ((d-1)-subs).
  ///
  /// The boundary faces are characterized by the fact that they
  /// only have 1 cell as super entity.
  pub fn boundary_faces(&self) -> Vec<SimplexHandle> {
    self
      .faces()
      .iter()
      .filter(|f| f.anti_boundary().len() == 1)
      .collect()
  }

  /// The nodes that lie on the boundary of the mesh.
  /// No particular order of nodes.
  pub fn boundary_nodes(&self) -> Vec<VertexIdx> {
    self
      .boundary_faces()
      .into_iter()
      .flat_map(|face| face.sorted_vertplex().iter().copied())
      .unique()
      .collect()
  }

  pub fn flag_boundary_nodes(&self) -> Vec<bool> {
    util::indicies_to_flags(&self.boundary_nodes(), self.nnodes())
  }

  pub fn boundary_cells(&self) -> Vec<CellIdx> {
    self
      .boundary_faces()
      .into_iter()
      // the boundary has only one parent cell by definition
      .map(|face| face.anti_boundary().idxs[0])
      .unique()
      .collect()
  }
}
