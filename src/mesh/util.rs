use super::{coordinates::NodeCoords, VertexIdx};

pub struct NodeData<T> {
  data: Vec<T>,
}
impl<T> NodeData<T> {
  pub fn new(data: Vec<T>) -> Self {
    Self { data }
  }
  pub fn from_coords_map<F>(coords: &NodeCoords, map: F) -> Self
  where
    F: FnMut(na::DVectorView<f64>) -> T,
  {
    let data = coords.coords().column_iter().map(map).collect();
    Self { data }
  }
}
impl<T> std::ops::Index<VertexIdx> for NodeData<T> {
  type Output = T;

  fn index(&self, index: VertexIdx) -> &Self::Output {
    &self.data[index]
  }
}
impl<T> std::ops::IndexMut<VertexIdx> for NodeData<T> {
  fn index_mut(&mut self, index: VertexIdx) -> &mut Self::Output {
    &mut self.data[index]
  }
}
