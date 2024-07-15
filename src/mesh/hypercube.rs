use super::{MeshSimplex, NodeId, SimplicialMesh};
use crate::{util::factorial, Dim};

use itertools::Itertools;

pub struct HyperRectangle {
  ndims: Dim,
  min: na::DVector<f64>,
  max: na::DVector<f64>,
}
impl HyperRectangle {
  pub fn new_min_max(min: na::DVector<f64>, max: na::DVector<f64>) -> Self {
    assert!(min.len() == max.len());
    let d = min.len();
    Self { ndims: d, min, max }
  }
  pub fn new_unit(ndims: Dim) -> Self {
    let min = na::DVector::zeros(ndims);
    let max = na::DVector::from_element(ndims, 1.0);
    Self { ndims, min, max }
  }
  pub fn new_uniscaled_unit(ndims: Dim, scale: f64) -> Self {
    let min = na::DVector::zeros(ndims);
    let max = na::DVector::from_element(ndims, scale);
    Self { ndims, min, max }
  }

  pub fn ndims(&self) -> usize {
    self.ndims
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

pub fn linear_idx2cartesian_idx(mut idx: usize, d: Dim, dlen: usize) -> na::DVector<usize> {
  let mut coord = na::DVector::zeros(d);
  for icomp in 0..d {
    coord[icomp] = idx % dlen;
    idx /= dlen;
  }
  coord
}

pub fn cartesian_idx2linear_idx(coord: na::DVector<usize>, dlen: usize) -> usize {
  let d = coord.len();
  let mut idx = 0;
  for icomp in (0..d).rev() {
    idx *= dlen;
    idx += coord[icomp];
  }
  idx
}

pub fn linear_idx2cartesian_coords(
  idx: usize,
  hypercube: &HyperRectangle,
  dlen: usize,
) -> na::DVector<f64> {
  (linear_idx2cartesian_idx(idx, hypercube.ndims(), dlen).cast::<f64>() / (dlen - 1) as f64)
    .component_mul(&hypercube.side_lengths())
    + hypercube.min()
}

pub fn hypercube_mesh_nodes(
  d: usize,
  nsubdivisions: usize,
  hypercube: &HyperRectangle,
) -> na::DMatrix<f64> {
  let nodes_per_dim = nsubdivisions + 1;
  let nnodes = nodes_per_dim.pow(d as u32);
  let mut nodes = na::DMatrix::zeros(d, nnodes);

  for (inode, mut coord) in nodes.column_iter_mut().enumerate() {
    coord.copy_from(&linear_idx2cartesian_coords(
      inode,
      hypercube,
      nodes_per_dim,
    ));
  }

  nodes
}

/// Create a structured mesh of the unit hypercube $[0, 1]^d$.
pub fn hypercube_mesh(hypercube: &HyperRectangle, nsubdivisions: usize) -> SimplicialMesh {
  let d = hypercube.ndims();
  let nodes_per_dim = nsubdivisions + 1;
  let ncubes = nsubdivisions.pow(d as u32);
  let nsimplicies = factorial(d) * ncubes;
  let mut simplicies = Vec::with_capacity(nsimplicies);

  for icube in 0..ncubes {
    let cube_coord = linear_idx2cartesian_idx(icube, d, nsubdivisions);

    simplicies.extend((0..d).permutations(d).map(|permut| {
      let mut vertices = Vec::with_capacity(d + 1);
      let mut vertex = cube_coord.clone();
      vertices.push(cartesian_idx2linear_idx(vertex.clone(), nodes_per_dim));
      for p in permut {
        vertex[p] += 1;
        vertices.push(cartesian_idx2linear_idx(vertex.clone(), nodes_per_dim));
      }
      MeshSimplex::new(vertices)
    }));
  }

  let nodes = hypercube_mesh_nodes(d, nsubdivisions, hypercube);
  SimplicialMesh::new(nodes, simplicies)
}

pub fn is_hypercube_node_on_boundary(mesh: &SimplicialMesh, node: NodeId) -> bool {
  let d = mesh.dim_intrinsic();
  let nodes_per_dim = mesh.nnodes() / d;
  let coords = linear_idx2cartesian_idx(node, d, nodes_per_dim);
  coords.iter().any(|&c| c == 0 || c == nodes_per_dim - 1)
}

#[cfg(test)]
mod test {
  use super::hypercube_mesh;
  use super::HyperRectangle;

  #[test]
  fn unit_cube_mesh() {
    let mesh = hypercube_mesh(&HyperRectangle::new_unit(3), 1);
    #[rustfmt::skip]
    let node_coords = na::DMatrix::from_column_slice(3, 8, &[
      0., 0., 0.,
      1., 0., 0.,
      0., 1., 0.,
      1., 1., 0.,
      0., 0., 1.,
      1., 0., 1.,
      0., 1., 1.,
      1., 1., 1.,
    ]);
    assert_eq!(*mesh.node_coords(), node_coords);
    let expected_simplicies = vec![
      &[0, 1, 3, 7],
      &[0, 1, 5, 7],
      &[0, 2, 3, 7],
      &[0, 2, 6, 7],
      &[0, 4, 5, 7],
      &[0, 4, 6, 7],
    ];
    let computed_simplicies: Vec<_> = mesh.cells().iter().map(|c| c.vertices()).collect();
    assert_eq!(computed_simplicies, expected_simplicies);
  }
}
