use crate::coordinates::MeshNodeCoords;

use super::NodeId;

/// This is generic data associated with each simplex of some particular dimension.
/// If the dimension matches the form rank, then this can be seen as DOF data.
/// It can be used for storing the boundary data.
// TODO!
//pub struct MeshData<T> {
//  dim: Dim,
//  data: Vec<T>,
//}

pub struct NodeData<T> {
  data: Vec<T>,
}
impl<T> NodeData<T> {
  pub fn from_coords_map<F>(coords: &MeshNodeCoords, map: F) -> Self
  where
    F: FnMut(na::DVectorView<f64>) -> T,
  {
    let data = coords.coords().column_iter().map(map).collect();
    Self { data }
  }
}
impl<T> std::ops::Index<NodeId> for NodeData<T> {
  type Output = T;

  fn index(&self, index: NodeId) -> &Self::Output {
    &self.data[index]
  }
}
impl<T> std::ops::IndexMut<NodeId> for NodeData<T> {
  fn index_mut(&mut self, index: NodeId) -> &mut Self::Output {
    &mut self.data[index]
  }
}
