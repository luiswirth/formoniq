use std::collections::HashSet;

use super::{NodeId, SimplexId, SimplicialMesh};

impl SimplicialMesh {
  /// For a d-mesh computes the boundary, which consists of (d-1)-simplicies.
  /// The boundary simplicies are characterized by the fact that they
  /// only have 1 super entity.
  pub fn boundary(&self) -> Vec<SimplexId> {
    let mut boundary = Vec::new();
    let boundary_dim = self.dim_intrinsic() - 1;
    for simp in self.dsimplicies(boundary_dim) {
      if simp.supers.len() == 1 {
        boundary.push(simp.simplex_id());
      }
    }
    boundary
  }

  /// The nodes that lie on the boundary of the mesh.
  /// No particular order of nodes.
  pub fn boundary_nodes(&self) -> Vec<NodeId> {
    let mut boundary_nodes = HashSet::new();
    let boundary = self.boundary();
    for boundary_simp in boundary {
      boundary_nodes.extend(self.simplex(boundary_simp).vertices().iter());
    }
    boundary_nodes.into_iter().collect()
  }
}
