use itertools::Itertools;

use super::{
  coordinates::{CoordManifold, NodeCoords},
  raw::RawSimplicialManifold,
  SimplicialManifold, VertexIdx,
};
use crate::{
  combinatorics::{factorial, OrderedSimplex, OrientedSimplex},
  util::{cartesian_index2linear_index, linear_index2cartesian_index},
  Dim,
};

pub struct HyperBox {
  min: na::DVector<f64>,
  max: na::DVector<f64>,
}

// constructors
impl HyperBox {
  pub fn new_min_max(min: na::DVector<f64>, max: na::DVector<f64>) -> Self {
    assert!(min.len() == max.len());
    Self { min, max }
  }
  pub fn new_unit(dim: Dim) -> Self {
    let min = na::DVector::zeros(dim);
    let max = na::DVector::from_element(dim, 1.0);
    Self { min, max }
  }
  pub fn new_unit_scaled(dim: Dim, scale: f64) -> Self {
    let min = na::DVector::zeros(dim);
    let max = na::DVector::from_element(dim, scale);
    Self { min, max }
  }
}

// getters
impl HyperBox {
  pub fn dim(&self) -> usize {
    self.min.len()
  }
  pub fn min(&self) -> na::DVector<f64> {
    self.min.clone()
  }
  pub fn max(&self) -> na::DVector<f64> {
    self.max.clone()
  }
  pub fn side_lengths(&self) -> na::DVector<f64> {
    &self.max - &self.min
  }
}

/// helper struct
pub struct HyperBoxMeshInfo {
  hyperbox: HyperBox,
  nboxes_per_dim: usize,
}
// constructors
impl HyperBoxMeshInfo {
  pub fn new_min_max(min: na::DVector<f64>, max: na::DVector<f64>, nboxes_per_dim: usize) -> Self {
    let hyperbox = HyperBox::new_min_max(min, max);
    Self {
      hyperbox,
      nboxes_per_dim,
    }
  }
  pub fn new_unit(dim: Dim, nboxes_per_dim: usize) -> Self {
    let hyperbox = HyperBox::new_unit(dim);
    Self {
      hyperbox,
      nboxes_per_dim,
    }
  }
  pub fn new_unit_scaled(dim: Dim, scale: f64, nboxes_per_dim: usize) -> Self {
    let hyperbox = HyperBox::new_unit_scaled(dim, scale);
    Self {
      hyperbox,
      nboxes_per_dim,
    }
  }
}
// getters
impl HyperBoxMeshInfo {
  pub fn hyperbox(&self) -> &HyperBox {
    &self.hyperbox
  }
  pub fn dim(&self) -> usize {
    self.hyperbox.dim()
  }
  pub fn min(&self) -> na::DVector<f64> {
    self.hyperbox.min()
  }
  pub fn max(&self) -> na::DVector<f64> {
    self.hyperbox.max()
  }
  pub fn side_lengths(&self) -> na::DVector<f64> {
    self.hyperbox.side_lengths()
  }
  pub fn nboxes_per_dim(&self) -> usize {
    self.nboxes_per_dim
  }
  pub fn nnodes_per_dim(&self) -> usize {
    self.nboxes_per_dim + 1
  }
  pub fn nboxes(&self) -> usize {
    self.nboxes_per_dim.pow(self.dim() as u32)
  }
  pub fn nnodes(&self) -> usize {
    self.nnodes_per_dim().pow(self.dim() as u32)
  }
  pub fn node_cart_idx(&self, inode: VertexIdx) -> na::DVector<usize> {
    linear_index2cartesian_index(inode, self.nnodes_per_dim(), self.dim())
  }
  pub fn node_pos(&self, inode: VertexIdx) -> na::DVector<f64> {
    (self.node_cart_idx(inode).cast::<f64>() / (self.nnodes_per_dim() - 1) as f64)
      .component_mul(&self.side_lengths())
      + self.min()
  }

  pub fn is_node_on_boundary(&self, node: VertexIdx) -> bool {
    let coords = linear_index2cartesian_index(node, self.nnodes_per_dim(), self.hyperbox.dim());
    coords
      .iter()
      .any(|&c| c == 0 || c == self.nnodes_per_dim() - 1)
  }

  pub fn boundary_nodes(&self) -> Vec<VertexIdx> {
    let mut r = Vec::new();
    for d in 0..self.dim() {
      let nnodes_boundary_face = self.nnodes_per_dim().pow(self.dim() as u32 - 1);
      for inode in 0..nnodes_boundary_face {
        let node_icart = linear_index2cartesian_index(inode, self.nnodes_per_dim(), self.dim() - 1);
        let low_boundary = node_icart.clone().insert_row(d, 0);
        let high_boundary = node_icart.insert_row(d, self.nnodes_per_dim() - 1);
        let low_boundary = cartesian_index2linear_index(low_boundary, self.dim());
        let high_boundary = cartesian_index2linear_index(high_boundary, self.dim());
        r.push(low_boundary);
        r.push(high_boundary);
      }
    }
    r
  }
}

impl HyperBoxMeshInfo {
  pub fn compute_node_coords(&self) -> NodeCoords {
    let mut nodes = na::DMatrix::zeros(self.dim(), self.nnodes());
    for (inode, mut coord) in nodes.column_iter_mut().enumerate() {
      coord.copy_from(&self.node_pos(inode));
    }
    NodeCoords::new(nodes)
  }

  pub fn compute_coord_manifold(&self) -> CoordManifold {
    let node_coords = self.compute_node_coords();

    let dim = self.dim();
    let ncells = factorial(dim) * self.nboxes();
    let mut cells: Vec<OrientedSimplex> = Vec::with_capacity(ncells);

    // iterate through all boxes that make up the mesh
    for icube in 0..self.nboxes() {
      let cube_icart = linear_index2cartesian_index(icube, self.nboxes_per_dim, self.dim());

      let vertex_icart_origin = cube_icart;
      let ivertex_origin =
        cartesian_index2linear_index(vertex_icart_origin.clone(), self.nnodes_per_dim());

      let basisdirs: Vec<_> = (0..dim).collect();

      // construct all $d!$ simplicies that make up the current box
      // each permutation of the basis directions (dimensions) gives rise to one simplicial cell
      let cube_cells = basisdirs
        .iter()
        .copied()
        // TODO: replace itertools with custom impl
        .permutations(basisdirs.len())
        .map(|basisdirs| {
          // construct simplex by adding all shifted vertices
          let mut cell = vec![ivertex_origin];

          // add every shift (according to permutation) to vertex iteratively
          // every shift step gives us one vertex
          let mut vertex_icart = vertex_icart_origin.clone();
          for basisdir in basisdirs {
            vertex_icart[basisdir] += 1;

            let ivertex = cartesian_index2linear_index(vertex_icart.clone(), self.nnodes_per_dim());
            cell.push(ivertex);
          }

          let cell = OrderedSimplex::new(cell);

          // Ensure consistent positive orientation of cells.
          // TODO: avoid computing orientation using coordinates / determinant.
          let orientation = node_coords.coord_simplex(&cell).orientation();

          OrientedSimplex::new(cell, orientation)
        });

      cells.extend(cube_cells);
    }

    CoordManifold::new(cells, node_coords)
  }

  pub fn compute_raw_manifold(&self) -> RawSimplicialManifold {
    self.compute_coord_manifold().into_raw_manifold()
  }
  pub fn compute_manifold(&self) -> SimplicialManifold {
    SimplicialManifold::new(self.compute_raw_manifold())
  }
}

#[cfg(test)]
mod test {
  use super::HyperBoxMeshInfo;

  #[test]
  fn unit_cube_mesh() {
    let mesh = HyperBoxMeshInfo::new_unit(3, 1).compute_coord_manifold();
    #[rustfmt::skip]
    let expected_nodes = na::DMatrix::from_column_slice(3, 8, &[
      0., 0., 0.,
      1., 0., 0.,
      0., 1., 0.,
      1., 1., 0.,
      0., 0., 1.,
      1., 0., 1.,
      0., 1., 1.,
      1., 1., 1.,
    ]);
    let computed_nodes = mesh.node_coords().coords();
    assert_eq!(*computed_nodes, expected_nodes);
    let expected_simplicies = vec![
      &[0, 1, 3, 7],
      &[0, 1, 5, 7],
      &[0, 2, 3, 7],
      &[0, 2, 6, 7],
      &[0, 4, 5, 7],
      &[0, 4, 6, 7],
    ];
    let computed_simplicies: Vec<_> = mesh
      .cells()
      .iter()
      .cloned()
      .map(|c| c.into_vertices())
      .collect();
    assert_eq!(computed_simplicies, expected_simplicies);
  }

  #[test]
  fn unit_square_mesh() {
    let mesh = HyperBoxMeshInfo::new_unit(2, 2).compute_coord_manifold();
    #[rustfmt::skip]
    let expected_nodes = na::DMatrix::from_column_slice(2, 9, &[
      0.0, 0.0,
      0.5, 0.0,
      1.0, 0.0,
      0.0, 0.5,
      0.5, 0.5,
      1.0, 0.5,
      0.0, 1.0,
      0.5, 1.0,
      1.0, 1.0,
    ]);
    let computed_nodes = mesh.node_coords().coords();
    assert_eq!(*computed_nodes, expected_nodes);
    let expected_simplicies = vec![
      &[0, 1, 4],
      &[0, 3, 4],
      &[1, 2, 5],
      &[1, 4, 5],
      &[3, 4, 7],
      &[3, 6, 7],
      &[4, 5, 8],
      &[4, 7, 8],
    ];
    let computed_simplicies: Vec<_> = mesh
      .cells()
      .iter()
      .cloned()
      .map(|c| c.into_vertices())
      .collect();
    assert_eq!(computed_simplicies, expected_simplicies);
  }
}
