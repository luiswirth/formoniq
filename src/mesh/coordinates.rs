use crate::{mesh::NodeId, Dim};

use std::rc::Rc;

#[derive(Debug, Clone)]
pub struct MeshNodeCoords {
  /// The coordinates of the nodes in the columns of a matrix.
  coords: na::DMatrix<f64>,
}
impl MeshNodeCoords {
  pub fn new(coords: na::DMatrix<f64>) -> Rc<Self> {
    Rc::new(Self { coords })
  }
  pub fn dim(&self) -> Dim {
    self.coords.nrows()
  }
  pub fn len(&self) -> usize {
    self.coords.ncols()
  }
  pub fn is_empty(&self) -> bool {
    self.len() == 0
  }
  pub fn coords(&self) -> &na::DMatrix<f64> {
    &self.coords
  }
  pub fn coord(&self, inode: NodeId) -> na::DVectorView<f64> {
    self.coords.column(inode)
  }
}
