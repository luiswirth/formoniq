use super::{MeshNodes, NodeId, SimplicialMesh};
use crate::{
  assemble::DofCoeffMap,
  space::DofId,
  util::{cartesian_index2linear_index, factorial, linear_index2cartesian_index},
  Dim,
};

use itertools::Itertools;
use std::rc::Rc;

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
  pub fn node_pos(&self, inode: NodeId) -> na::DVector<f64> {
    (linear_index2cartesian_index(inode, self.nnodes_per_dim(), self.dim()).cast::<f64>()
      / (self.nnodes_per_dim() - 1) as f64)
      .component_mul(&self.side_lengths())
      + self.min()
  }

  pub fn is_node_on_boundary(&self, node: NodeId) -> bool {
    let coords = linear_index2cartesian_index(node, self.nnodes_per_dim(), self.hyperbox.dim());
    coords
      .iter()
      .any(|&c| c == 0 || c == self.nnodes_per_dim() - 1)
  }

  pub fn boundary_nodes(&self) -> Vec<NodeId> {
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

pub struct HyperBoxMesh {
  info: HyperBoxMeshInfo,
  nodes: Rc<MeshNodes>,
  mesh: Rc<SimplicialMesh>,
}

// constructors
impl HyperBoxMesh {
  pub fn new_min_max(min: na::DVector<f64>, max: na::DVector<f64>, nboxes_per_dim: usize) -> Self {
    let info = HyperBoxMeshInfo::new_min_max(min, max, nboxes_per_dim);
    Self::from_info(info)
  }
  pub fn new_unit(dim: Dim, nboxes_per_dim: usize) -> Self {
    let info = HyperBoxMeshInfo::new_unit(dim, nboxes_per_dim);
    Self::from_info(info)
  }
  pub fn new_unit_scaled(dim: Dim, scale: f64, nboxes_per_dim: usize) -> Self {
    let info = HyperBoxMeshInfo::new_unit_scaled(dim, scale, nboxes_per_dim);
    Self::from_info(info)
  }

  pub fn from_info(info: HyperBoxMeshInfo) -> Self {
    let nodes = Self::compute_mesh_nodes(&info);
    let mesh = Self::compute_mesh(&info, nodes.clone());
    Self { info, nodes, mesh }
  }
}

// getters
impl HyperBoxMesh {
  pub fn hyperbox(&self) -> &HyperBox {
    self.info.hyperbox()
  }
  pub fn info(&self) -> &HyperBoxMeshInfo {
    &self.info
  }
  pub fn nodes(&self) -> &Rc<MeshNodes> {
    &self.nodes
  }
  pub fn mesh(&self) -> &Rc<SimplicialMesh> {
    &self.mesh
  }

  pub fn dim(&self) -> usize {
    self.info.dim()
  }
  pub fn min(&self) -> na::DVector<f64> {
    self.info.min()
  }
  pub fn max(&self) -> na::DVector<f64> {
    self.info.max()
  }
  pub fn side_lengths(&self) -> na::DVector<f64> {
    self.info.side_lengths()
  }
  pub fn nboxes_per_dim(&self) -> usize {
    self.info.nboxes_per_dim()
  }
  pub fn nnodes_per_dim(&self) -> usize {
    self.info.nnodes_per_dim()
  }
  pub fn nnodes(&self) -> usize {
    self.info.nnodes()
  }
  pub fn nboxes(&self) -> usize {
    self.info.nboxes()
  }
  pub fn is_node_on_boundary(&self, node: NodeId) -> bool {
    self.info.is_node_on_boundary(node)
  }
  pub fn boundary_nodes(&self) -> Vec<NodeId> {
    self.info.boundary_nodes()
  }
}

// construction helpers
impl HyperBoxMesh {
  fn compute_mesh_nodes(info: &HyperBoxMeshInfo) -> Rc<MeshNodes> {
    let mut nodes = na::DMatrix::zeros(info.dim(), info.nnodes());
    for (inode, mut coord) in nodes.column_iter_mut().enumerate() {
      coord.copy_from(&info.node_pos(inode));
    }
    MeshNodes::new(nodes)
  }

  fn compute_mesh(info: &HyperBoxMeshInfo, nodes: Rc<MeshNodes>) -> Rc<SimplicialMesh> {
    let nsimplicies = factorial(info.dim()) * info.nboxes();
    let mut simplicies = Vec::with_capacity(nsimplicies);

    for icube in 0..info.nboxes() {
      let cube_coord = linear_index2cartesian_index(icube, info.nboxes_per_dim, info.dim());

      simplicies.extend((0..info.dim()).permutations(info.dim()).map(|permut| {
        let mut vertices = Vec::with_capacity(info.dim() + 1);
        let mut vertex = cube_coord.clone();
        vertices.push(cartesian_index2linear_index(
          vertex.clone(),
          info.nnodes_per_dim(),
        ));
        for p in permut {
          vertex[p] += 1;
          vertices.push(cartesian_index2linear_index(
            vertex.clone(),
            info.nnodes_per_dim(),
          ));
        }
        vertices
      }));
    }

    SimplicialMesh::from_cells(nodes, simplicies)
  }
}

#[derive(Clone)]
pub struct HyperBoxDirichletBcMap<'a, F>
where
  F: Fn(na::DVectorView<f64>) -> f64,
{
  mesh: &'a HyperBoxMesh,
  dirichlet_data: F,
}
impl<F> DofCoeffMap for HyperBoxDirichletBcMap<'_, F>
where
  F: Fn(na::DVectorView<f64>) -> f64,
{
  fn eval(&self, idof: DofId) -> Option<f64> {
    self.mesh.is_node_on_boundary(idof).then(|| {
      let pos = self.mesh.nodes.coord(idof);
      (self.dirichlet_data)(pos)
    })
  }
}
impl<'a, F> HyperBoxDirichletBcMap<'a, F>
where
  F: Fn(na::DVectorView<f64>) -> f64,
{
  pub fn new(mesh: &'a HyperBoxMesh, dirichlet_data: F) -> Self {
    Self {
      dirichlet_data,
      mesh,
    }
  }
}

#[cfg(test)]
mod test {
  use super::HyperBoxMesh;

  #[test]
  fn unit_cube_mesh() {
    let mesh = HyperBoxMesh::new_unit(3, 1);
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
    assert_eq!(*mesh.nodes().coords(), node_coords);
    let expected_simplicies = vec![
      &[0, 1, 3, 7],
      &[0, 1, 5, 7],
      &[0, 2, 3, 7],
      &[0, 2, 6, 7],
      &[0, 4, 5, 7],
      &[0, 4, 6, 7],
    ];
    let computed_simplicies: Vec<_> = mesh.mesh().cells().iter().map(|c| c.vertices()).collect();
    assert_eq!(computed_simplicies, expected_simplicies);
  }
}
