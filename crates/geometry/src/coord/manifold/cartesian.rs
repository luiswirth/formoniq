use crate::coord::MeshVertexCoords;

use index_algebra::{factorial, IndexSet};
use topology::{
  complex::TopologyComplex,
  simplex::{Simplex, SortedSimplex},
  skeleton::TopologySkeleton,
  Dim,
};

/// converts linear index to cartesian index
///
/// converts linear index in 0..dim_len^d to cartesian index in (0)^d..(dim_len)^d
pub fn linear_index2cartesian_index(
  mut lin_idx: usize,
  dim_len: usize,
  dim: usize,
) -> na::DVector<usize> {
  let mut cart_idx = na::DVector::zeros(dim);
  for icomp in 0..dim {
    cart_idx[icomp] = lin_idx % dim_len;
    lin_idx /= dim_len;
  }
  cart_idx
}

/// converts cartesian index to linear index
///
/// converts cartesian index in (0)^d..(dim_len)^d to linear index in 0..dim_len^d
pub fn cartesian_index2linear_index(cart_idx: na::DVector<usize>, dim_len: usize) -> usize {
  let dim = cart_idx.len();
  let mut lin_idx = 0;
  for icomp in (0..dim).rev() {
    lin_idx *= dim_len;
    lin_idx += cart_idx[icomp];
  }
  lin_idx
}

pub struct Rect {
  min: na::DVector<f64>,
  max: na::DVector<f64>,
}

impl Rect {
  pub fn new_min_max(min: na::DVector<f64>, max: na::DVector<f64>) -> Self {
    assert!(min.len() == max.len());
    Self { min, max }
  }
  pub fn new_unit_cube(dim: Dim) -> Self {
    let min = na::DVector::zeros(dim);
    let max = na::DVector::from_element(dim, 1.0);
    Self { min, max }
  }
  pub fn new_scaled_cube(dim: Dim, scale: f64) -> Self {
    let min = na::DVector::zeros(dim);
    let max = na::DVector::from_element(dim, scale);
    Self { min, max }
  }

  pub fn dim(&self) -> usize {
    self.min.len()
  }
  pub fn min(&self) -> &na::DVector<f64> {
    &self.min
  }
  pub fn max(&self) -> &na::DVector<f64> {
    &self.max
  }
  pub fn side_lengths(&self) -> na::DVector<f64> {
    &self.max - &self.min
  }
}

pub struct CartesianMesh {
  rect: Rect,
  ncells_axis: usize,
}
// constructors
impl CartesianMesh {
  pub fn new_min_max(min: na::DVector<f64>, max: na::DVector<f64>, ncells_axis: usize) -> Self {
    let rect = Rect::new_min_max(min, max);
    Self { rect, ncells_axis }
  }
  pub fn new_unit(dim: Dim, ncells_axis: usize) -> Self {
    let rect = Rect::new_unit_cube(dim);
    Self { rect, ncells_axis }
  }
  pub fn new_unit_scaled(dim: Dim, ncells_axis: usize, scale: f64) -> Self {
    let rect = Rect::new_scaled_cube(dim, scale);
    Self { rect, ncells_axis }
  }
}
// getters
impl CartesianMesh {
  pub fn rect(&self) -> &Rect {
    &self.rect
  }
  pub fn dim(&self) -> usize {
    self.rect.dim()
  }
  pub fn min(&self) -> &na::DVector<f64> {
    self.rect.min()
  }
  pub fn max(&self) -> &na::DVector<f64> {
    self.rect.max()
  }
  pub fn side_lengths(&self) -> na::DVector<f64> {
    self.rect.side_lengths()
  }
  pub fn ncells_axis(&self) -> usize {
    self.ncells_axis
  }
  pub fn nvertices_axis(&self) -> usize {
    self.ncells_axis + 1
  }
  pub fn ncells(&self) -> usize {
    self.ncells_axis.pow(self.dim() as u32)
  }
  pub fn nvertices(&self) -> usize {
    self.nvertices_axis().pow(self.dim() as u32)
  }
  pub fn vertex_cart_idx(&self, ivertex: usize) -> na::DVector<usize> {
    linear_index2cartesian_index(ivertex, self.nvertices_axis(), self.dim())
  }
  pub fn vertex_pos(&self, ivertex: usize) -> na::DVector<f64> {
    (self.vertex_cart_idx(ivertex).cast::<f64>() / (self.nvertices_axis() - 1) as f64)
      .component_mul(&self.side_lengths())
      + self.min()
  }

  pub fn is_vertex_on_boundary(&self, vertex: usize) -> bool {
    let coords = linear_index2cartesian_index(vertex, self.nvertices_axis(), self.rect.dim());
    coords
      .iter()
      .any(|&c| c == 0 || c == self.nvertices_axis() - 1)
  }

  pub fn boundary_vertices(&self) -> Vec<usize> {
    let mut r = Vec::new();
    for d in 0..self.dim() {
      let nvertices_boundary_face = self.nvertices_axis().pow(self.dim() as u32 - 1);
      for ivertex in 0..nvertices_boundary_face {
        let vertex_icart =
          linear_index2cartesian_index(ivertex, self.nvertices_axis(), self.dim() - 1);
        let low_boundary = vertex_icart.clone().insert_row(d, 0);
        let high_boundary = vertex_icart.insert_row(d, self.nvertices_axis() - 1);
        let low_boundary = cartesian_index2linear_index(low_boundary, self.nvertices_axis());
        let high_boundary = cartesian_index2linear_index(high_boundary, self.nvertices_axis());
        r.push(low_boundary);
        r.push(high_boundary);
      }
    }
    r
  }
}

impl CartesianMesh {
  pub fn compute_coord_complex(&self) -> (TopologyComplex, MeshVertexCoords) {
    let (skeleton, coords) = self.compute_coord_facets();
    let complex = TopologyComplex::from_facet_skeleton(skeleton);
    (complex, coords)
  }

  pub fn compute_coord_facets(&self) -> (TopologySkeleton, MeshVertexCoords) {
    let mut coords = na::DMatrix::zeros(self.dim(), self.nvertices());
    for (ivertex, mut coord) in coords.column_iter_mut().enumerate() {
      coord.copy_from(&self.vertex_pos(ivertex));
    }
    let coords = MeshVertexCoords::new(coords);

    let nboxes = self.ncells();
    let nboxes_axis = self.ncells_axis();

    let dim = self.dim();
    let nsimplicies = factorial(dim) * nboxes;
    let mut simplicies: Vec<SortedSimplex> = Vec::with_capacity(nsimplicies);

    // iterate through all boxes that make up the mesh
    for ibox in 0..nboxes {
      let cube_icart = linear_index2cartesian_index(ibox, nboxes_axis, self.dim());

      let vertex_icart_origin = cube_icart;
      let ivertex_origin =
        cartesian_index2linear_index(vertex_icart_origin.clone(), self.nvertices_axis());

      let basisdirs = IndexSet::increasing(dim);

      // Construct all $d!$ simplexes that make up the current box.
      // Each permutation of the basis directions (dimensions) gives rise to one simplex.
      let cube_simplicies = basisdirs.permutations().map(|basisdirs| {
        // Construct simplex by adding all shifted vertices.
        let mut simplex = vec![ivertex_origin];

        // Add every shift (according to permutation) to vertex iteratively.
        // Every shift step gives us one vertex.
        let mut vertex_icart = vertex_icart_origin.clone();
        for basisdir in basisdirs.set.iter() {
          vertex_icart[basisdir] += 1;

          let ivertex = cartesian_index2linear_index(vertex_icart.clone(), self.nvertices_axis());
          simplex.push(ivertex);
        }

        Simplex::from(simplex).assume_sorted()
      });

      simplicies.extend(cube_simplicies);
    }

    let skeleton = TopologySkeleton::new(simplicies);
    (skeleton, coords)
  }
}

#[cfg(test)]
mod test {
  use super::CartesianMesh;

  #[test]
  fn unit_cube_mesh() {
    let (mesh, coords) = CartesianMesh::new_unit(3, 1).compute_coord_facets();

    #[rustfmt::skip]
    let expected_coords = na::DMatrix::from_column_slice(3, 8, &[
      0., 0., 0.,
      1., 0., 0.,
      0., 1., 0.,
      1., 1., 0.,
      0., 0., 1.,
      1., 0., 1.,
      0., 1., 1.,
      1., 1., 1.,
    ]);
    assert_eq!(*coords.matrix(), expected_coords);

    let expected_facets = vec![
      &[0, 1, 3, 7],
      &[0, 1, 5, 7],
      &[0, 2, 3, 7],
      &[0, 2, 6, 7],
      &[0, 4, 5, 7],
      &[0, 4, 6, 7],
    ];
    let facets: Vec<_> = mesh
      .simplex_iter()
      .map(|s| s.vertices.clone().into_vec())
      .collect();
    assert_eq!(facets, expected_facets);
  }

  #[test]
  fn unit_square_mesh() {
    let (mesh, coords) = CartesianMesh::new_unit(2, 2).compute_coord_facets();

    #[rustfmt::skip]
    let expected_coords = na::DMatrix::from_column_slice(2, 9, &[
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
    assert_eq!(*coords.matrix(), expected_coords);

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
    let facets: Vec<_> = mesh
      .simplex_iter()
      .map(|s| s.vertices.clone().into_vec())
      .collect();
    assert_eq!(facets, expected_simplicies);
  }
}
