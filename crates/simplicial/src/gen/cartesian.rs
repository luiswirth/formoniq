use formoniq_linalg::nalgebra::{Matrix, Vector};
use itertools::Itertools;
use multiindex::{
  cartesian::{corner_offset, strides},
  factorial, Combination,
};

pub use multiindex::cartesian::{cartesian_index2linear_index, linear_index2cartesian_index};

use crate::{
  geometry::coord::mesh::MeshCoords,
  topology::{complex::Complex, simplex::Simplex, skeleton::Skeleton},
  Dim,
};

pub struct Rect {
  min: Vector,
  max: Vector,
}

impl Rect {
  pub fn new_min_max(min: Vector, max: Vector) -> Self {
    assert_eq!(min.len(), max.len());
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
  pub fn vertex_cart_idx(&self, ivertex: usize) -> Vec<usize> {
    linear_index2cartesian_index(ivertex, self.nvertices_axis(), self.dim())
  }
  pub fn vertex_pos(&self, ivertex: usize) -> Vector {
    let cart_idx = self.vertex_cart_idx(ivertex);
    let cart_idx = Vector::from_iterator(cart_idx.len(), cart_idx.iter().map(|&c| c as f64));
    (cart_idx / (self.nvertices_axis() - 1) as f64).component_mul(&self.side_lengths()) + self.min()
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
        let mut low_boundary = vertex_icart.clone();
        low_boundary.insert(d, 0);
        let mut high_boundary = vertex_icart;
        high_boundary.insert(d, self.nvertices_axis() - 1);
        let low_boundary = cartesian_index2linear_index(&low_boundary, self.nvertices_axis());
        let high_boundary = cartesian_index2linear_index(&high_boundary, self.nvertices_axis());
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

  /// Kuhn (Freudenthal) triangulation.
  ///
  /// The corners of a cube are the subsets of the axes (radix-2 cartesian
  /// indices are `Combination` bitsets), and each of the $d!$ simplices of a
  /// cube is a maximal chain
  /// $emptyset subset {a_1} subset {a_1, a_2} subset dots.c$
  /// in this subset lattice, one per permutation of the axes.
  pub fn compute_cell_skeleton(&self) -> Skeleton {
    let dim = self.dim();
    let strides = strides(self.nvertices_axis(), dim);

    let mut simplices: Vec<Simplex> = Vec::with_capacity(factorial(dim) * self.ncells());
    for ibox in 0..self.ncells() {
      let box_cart = linear_index2cartesian_index(ibox, self.ncells_axis(), dim);
      let origin = cartesian_index2linear_index(&box_cart, self.nvertices_axis());

      for axes in (0..dim).permutations(dim) {
        let chain = axes.iter().scan(Combination::empty(), |corner, &axis| {
          *corner = corner.inserted(axis);
          Some(*corner)
        });
        let vertices = std::iter::once(origin)
          .chain(chain.map(|corner| origin + corner_offset(corner, &strides)))
          .collect();
        simplices.push(Simplex::new(vertices));
      }
    }

    Skeleton::new(simplices)
  }
}

#[cfg(test)]
mod test {
  use super::CartesianMeshInfo;
  use formoniq_linalg::nalgebra::Matrix;

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

    // Cells in canonical colexicographic order.
    let expected_cells = vec![
      &[0, 1, 3, 7],
      &[0, 2, 3, 7],
      &[0, 1, 5, 7],
      &[0, 4, 5, 7],
      &[0, 2, 6, 7],
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

    let expected_simplices = vec![
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
    assert_eq!(cells, expected_simplices);
  }
}
