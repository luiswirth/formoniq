use common::Dim;

use crate::regge::EdgeLengths;

pub type VertexIdx = usize;

#[derive(Debug, Clone)]
pub struct VertexCoords {
  matrix: na::DMatrix<f64>,
}
impl VertexCoords {
  pub fn new(matrix: na::DMatrix<f64>) -> Self {
    Self { matrix }
  }
  pub fn dim(&self) -> Dim {
    self.matrix.nrows()
  }
  pub fn nvertices(&self) -> usize {
    self.matrix.ncols()
  }

  pub fn coord(&self, ivertex: VertexIdx) -> na::DVectorView<f64> {
    self.matrix.column(ivertex)
  }

  pub fn matrix(&self) -> &na::DMatrix<f64> {
    &self.matrix
  }
  pub fn matrix_mut(&mut self) -> &mut na::DMatrix<f64> {
    &mut self.matrix
  }
  pub fn into_matrix(self) -> na::DMatrix<f64> {
    self.matrix
  }

  pub fn eval_coord_fn<F>(&self, f: F) -> na::DVector<f64>
  where
    F: FnMut(na::DVectorView<f64>) -> f64,
  {
    na::DVector::from_iterator(self.nvertices(), self.matrix.column_iter().map(f))
  }

  pub fn embed_euclidean(mut self, dim: Dim) -> VertexCoords {
    let old_dim = self.matrix.nrows();
    self.matrix = self.matrix.insert_rows(old_dim, dim - old_dim, 0.0);
    self
  }

  pub fn to_edge_lengths(
    &self,
    edges: impl ExactSizeIterator<Item = [VertexIdx; 2]>,
  ) -> EdgeLengths {
    let mut edge_lengths = na::DVector::zeros(edges.len());
    for (iedge, edge) in edges.enumerate() {
      let [vi, vj] = edge;
      let length = (self.coord(vj) - self.coord(vi)).norm();
      edge_lengths[iedge] = length;
    }
    EdgeLengths::new(edge_lengths)
  }
}
