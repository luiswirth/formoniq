use super::{
  raw::{RawManifoldGeometry, RawManifoldTopology, RawSimplexTopology, RawSimplicialManifold},
  SimplicialManifold, SortedSimplex,
};
use crate::{mesh::VertexIdx, Dim};

use std::collections::{hash_map, HashMap};

#[derive(Debug, Clone)]
pub struct MeshNodeCoords {
  /// The coordinates of the nodes in the columns of a matrix.
  coords: na::DMatrix<f64>,
}
impl MeshNodeCoords {
  pub fn new(coords: na::DMatrix<f64>) -> Self {
    Self { coords }
  }
  pub fn dim(&self) -> Dim {
    self.coords.nrows()
  }
  pub fn nnodes(&self) -> usize {
    self.coords.ncols()
  }
  pub fn is_empty(&self) -> bool {
    self.nnodes() == 0
  }
  pub fn coords(&self) -> &na::DMatrix<f64> {
    &self.coords
  }
  pub fn coord(&self, inode: VertexIdx) -> na::DVectorView<f64> {
    self.coords.column(inode)
  }
}

pub struct CoordManifold {
  /// topology
  cells: Vec<RawSimplexTopology>,
  /// geometry
  node_coords: MeshNodeCoords,
}
impl CoordManifold {
  pub fn new(cells: Vec<RawSimplexTopology>, node_coords: MeshNodeCoords) -> Self {
    Self { cells, node_coords }
  }

  pub fn into_raw_manifold(self) -> RawSimplicialManifold {
    let mut edge_lengths = HashMap::new();

    for cell in &self.cells {
      for &v0 in &cell.vertices {
        for &v1 in &cell.vertices {
          let edge = SortedSimplex::edge(v0, v1);
          if let hash_map::Entry::Vacant(e) = edge_lengths.entry(edge) {
            let length = (self.node_coords.coord(v1) - self.node_coords.coord(v0)).norm();
            e.insert(length);
          }
        }
      }
    }

    RawSimplicialManifold::new(
      self.node_coords.nnodes(),
      RawManifoldTopology::new(self.cells),
      RawManifoldGeometry::new(edge_lengths),
    )
  }

  pub fn into_manifold(self) -> SimplicialManifold {
    SimplicialManifold::from_raw(self.into_raw_manifold())
  }
}

impl CoordManifold {
  pub fn cells(&self) -> &[RawSimplexTopology] {
    &self.cells
  }
  pub fn node_coords(&self) -> &MeshNodeCoords {
    &self.node_coords
  }
}
