use crate::util;

use super::{CellIdx, SimplexHandle, SimplicialManifold, VertexIdx};

use itertools::Itertools;

impl SimplicialManifold {
  pub fn has_boundary(&self) -> bool {
    !self.boundary_facets().is_empty()
  }

  /// For a d-mesh computes the boundary, which consists of facets ((d-1)-faces).
  ///
  /// The boundary facets are characterized by the fact that they
  /// only have 1 cell as super entity.
  pub fn boundary_facets(&self) -> Vec<SimplexHandle> {
    self
      .facets()
      .iter()
      .filter(|f| f.antiboundary().len() == 1)
      .collect()
  }

  /// The nodes that lie on the boundary of the mesh.
  /// No particular order of nodes.
  pub fn boundary_nodes(&self) -> Vec<VertexIdx> {
    self
      .boundary_facets()
      .into_iter()
      .flat_map(|face| face.canonical_vertplex().iter().copied())
      .unique()
      .collect()
  }

  pub fn flag_boundary_nodes(&self) -> Vec<bool> {
    util::indicies_to_flags(&self.boundary_nodes(), self.nnodes())
  }

  pub fn boundary_cells(&self) -> Vec<CellIdx> {
    self
      .boundary_facets()
      .into_iter()
      // the boundary has only one super by definition
      .map(|face| face.antiboundary().iter().next().unwrap())
      .map(|cell| cell.kidx())
      .unique()
      .collect()
  }
}
