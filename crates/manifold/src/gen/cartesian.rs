use common::{
  combo::factorial,
  linalg::nalgebra::{Matrix, Vector},
};
use itertools::Itertools;

use crate::{
  geometry::coord::mesh::MeshCoords,
  topology::{complex::Complex, simplex::Simplex, skeleton::Skeleton},
  Dim,
};

/// converts linear index to cartesian index
///
/// converts linear index in 0..dim_len^d to cartesian index in (0)^d..(dim_len)^d
pub fn linear_index2cartesian_index(
  mut lin_idx: usize,
  dim_len: usize,
  dim: usize,
) -> Vector<usize> {
  let mut cart_idx = Vector::zeros(dim);
  for icomp in 0..dim {
    cart_idx[icomp] = lin_idx % dim_len;
    lin_idx /= dim_len;
  }
  cart_idx
}

/// converts cartesian index to linear index
///
/// converts cartesian index in (0)^d..(dim_len)^d to linear index in 0..dim_len^d
pub fn cartesian_index2linear_index(cart_idx: Vector<usize>, dim_len: usize) -> usize {
  let dim = cart_idx.len();
  let mut lin_idx = 0;
  for icomp in (0..dim).rev() {
    lin_idx *= dim_len;
    lin_idx += cart_idx[icomp];
  }
  lin_idx
}

pub struct Rect {
  min: Vector,
  max: Vector,
}

impl Rect {
  pub fn new_min_max(min: Vector, max: Vector) -> Self {
    assert!(min.len() == max.len());
    Self { min, max }
  }
  pub fn new_unit_cube(dim: Dim) -> Self {
    let min = Vector::zeros(dim);
    let max = Vector::from_element(dim, 1.0);
    Self { min, max }
  }
  pub fn new_scaled_cube(dim: Dim, scale: f64) -> Self {
    let min = Vector::zeros(dim);
    let max = Vector::from_element(dim, scale);
    Self { min, max }
  }

  pub fn dim(&self) -> usize {
    self.min.len()
  }
  pub fn min(&self) -> &Vector {
    &self.min
  }
  pub fn max(&self) -> &Vector {
    &self.max
  }
  pub fn side_lengths(&self) -> Vector {
    &self.max - &self.min
  }
}

pub struct CartesianMeshInfo {
  rect: Rect,
  ncells_axis: usize,
}
// constructors
impl CartesianMeshInfo {
  pub fn new_min_max(min: Vector, max: Vector, ncells_axis: usize) -> Self {
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
impl CartesianMeshInfo {
  pub fn rect(&self) -> &Rect {
    &self.rect
  }
  pub fn dim(&self) -> usize {
    self.rect.dim()
  }
  pub fn min(&self) -> &Vector {
    self.rect.min()
  }
  pub fn max(&self) -> &Vector {
    self.rect.max()
  }
  pub fn side_lengths(&self) -> Vector {
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
  pub fn vertex_cart_idx(&self, ivertex: usize) -> Vector<usize> {
    linear_index2cartesian_index(ivertex, self.nvertices_axis(), self.dim())
  }
  pub fn vertex_pos(&self, ivertex: usize) -> Vector {
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
      let nvertices_boundary_facet = self.nvertices_axis().pow(self.dim() as u32 - 1);
      for ivertex in 0..nvertices_boundary_facet {
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

impl CartesianMeshInfo {
  pub fn compute_coord_complex(&self) -> (Complex, MeshCoords) {
    let (skeleton, coords) = self.compute_coord_cells();
    let complex = Complex::from_cells(skeleton);
    (complex, coords)
  }
  pub fn compute_coord_cells(&self) -> (Skeleton, MeshCoords) {
    let skeleton = self.compute_cell_skeleton();
    let coords = self.compute_vertex_coords();
    (skeleton, coords)
  }
  pub fn compute_vertex_coords(&self) -> MeshCoords {
    let mut coords = Matrix::zeros(self.dim(), self.nvertices());
    for (ivertex, mut coord) in coords.column_iter_mut().enumerate() {
      coord.copy_from(&self.vertex_pos(ivertex));
    }
    MeshCoords::new(coords)
  }

  pub fn compute_cell_skeleton(&self) -> Skeleton {
    let nboxes = self.ncells();
    let nboxes_axis = self.ncells_axis();

    let dim = self.dim();
    let nsimplicies = factorial(dim) * nboxes;
    let mut simplicies: Vec<Simplex> = Vec::with_capacity(nsimplicies);

    // iterate through all boxes that make up the mesh
    for ibox in 0..nboxes {
      let cube_icart = linear_index2cartesian_index(ibox, nboxes_axis, self.dim());

      let vertex_icart_origin = cube_icart;
      let ivertex_origin =
        cartesian_index2linear_index(vertex_icart_origin.clone(), self.nvertices_axis());

      // Construct all $d!$ simplexes that make up the current box.
      // Each permutation of the basis directions (dimensions) gives rise to one simplex.
      let cube_simplicies = (0..dim).permutations(dim).map(|basisdirs| {
        // Construct simplex by adding all shifted vertices.
        let mut simplex = vec![ivertex_origin];

        // Add every shift (according to permutation) to vertex iteratively.
        // Every shift step gives us one vertex.
        let mut vertex_icart = vertex_icart_origin.clone();
        for &basisdir in basisdirs.iter() {
          vertex_icart[basisdir] += 1;

          let ivertex = cartesian_index2linear_index(vertex_icart.clone(), self.nvertices_axis());
          simplex.push(ivertex);
        }

        let simplex = Simplex::from(simplex);
        assert!(simplex.is_sorted());
        simplex
      });

      simplicies.extend(cube_simplicies);
    }

    Skeleton::new(simplicies)
  }
}

#[cfg(test)]
mod test {
  use super::CartesianMeshInfo;
  use common::linalg::nalgebra::Matrix;

  #[test]
  fn unit_cube_mesh() {
    let (mesh, coords) = CartesianMeshInfo::new_unit(3, 1).compute_coord_cells();

    #[rustfmt::skip]
    let expected_coords = Matrix::from_column_slice(3, 8, &[
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

    let expected_cells = vec![
      &[0, 1, 3, 7],
      &[0, 1, 5, 7],
      &[0, 2, 3, 7],
      &[0, 2, 6, 7],
      &[0, 4, 5, 7],
      &[0, 4, 6, 7],
    ];
    let cells: Vec<_> = mesh.into_iter().map(|s| s.vertices).collect();
    assert_eq!(cells, expected_cells);
  }

  #[test]
  fn unit_square_mesh() {
    let (mesh, coords) = CartesianMeshInfo::new_unit(2, 2).compute_coord_cells();

    #[rustfmt::skip]
    let expected_coords = Matrix::from_column_slice(2, 9, &[
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
    let cells: Vec<_> = mesh.into_iter().map(|s| s.vertices).collect();
    assert_eq!(cells, expected_simplicies);
  }
}
