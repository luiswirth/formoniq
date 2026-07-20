use crate::linalg::{Matrix, Vector};
use itertools::Itertools;
use multiindex::{
  cartesian::{cartesian2linear_mixed, corner_offset, linear2cartesian_mixed, mixed_strides},
  factorial, Combination,
};

pub use multiindex::cartesian::{cartesian2linear, linear2cartesian};

use crate::{
  geometry::coord::mesh::MeshCoords,
  topology::{complex::Complex, simplex::Simplex, skeleton::Skeleton},
  Dim,
};
use gramian::Gramian;

/// Time-axis scale of a causally generic Minkowski box (axis $0$ is time).
///
/// A Kuhn edge of the box spans a set $S$ of axes; under the Minkowski metric
/// its signed squared length is $s = -rho^2 [t in S] + abs(S sect "space")$,
/// null exactly when $t in S$ and $abs(S sect "space") = rho^2$. So no edge is
/// lightlike iff $rho^2$ is not an integer in $[0, "dim")$ -- and $rho = 0.7$
/// ($rho^2 = 0.49$) misses that integer set in every dimension. Uniform
/// refinement scales every edge by the same factor, so a refinement tower stays
/// causally generic. See [`CartesianGrid::minkowski`].
pub const CAUSAL_TIME_SCALE: f64 = 0.7;

/// An axis-aligned box $[min, max] subset RR^d$.
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

/// A structured grid on an axis-aligned box: a cell count *per axis*, their
/// product many boxes in all, each Kuhn-triangulated into $d!$ simplices.
/// [`triangulate`](Self::triangulate) produces the simplicial `Complex` and its
/// vertex `MeshCoords`.
///
/// The resolution is per-axis because the box's sides are: one count over a
/// box whose sides differ produces cells as anisotropic as the box, and the
/// shape regularity a mesh is judged by is a property of the *spacing*, not of
/// the count. A cube meshed with one count everywhere is the isotropic special
/// case, and [`Self::new_unit`] and friends are that case named.
pub struct CartesianGrid {
  rect: Rect,
  ncells: Vec<usize>,
}
// constructors
impl CartesianGrid {
  /// A grid with an independent cell count per axis.
  pub fn new_anisotropic(min: Vector, max: Vector, ncells: Vec<usize>) -> Self {
    let rect = Rect::new_min_max(min, max);
    assert_eq!(
      ncells.len(),
      rect.dim(),
      "One cell count per axis is required."
    );
    assert!(ncells.iter().all(|&n| n >= 1), "An axis needs a cell.");
    Self { rect, ncells }
  }
  /// A grid whose cells are as near cubical as the counts allow: the count on
  /// each axis is scaled by that axis's length, so the spacing is
  /// quasi-uniform. `ncells_longest` fixes the resolution of the longest axis.
  ///
  /// This is what keeps a long, thin box from being meshed into slivers, and it
  /// agrees with the isotropic constructors on a cube.
  pub fn new_quasi_uniform(min: Vector, max: Vector, ncells_longest: usize) -> Self {
    let rect = Rect::new_min_max(min, max);
    let sides = rect.side_lengths();
    let longest = sides.iter().copied().fold(0.0_f64, f64::max);
    let ncells = sides
      .iter()
      .map(|&side| ((side / longest * ncells_longest as f64).round() as usize).max(1))
      .collect();
    Self { rect, ncells }
  }
  pub fn new_min_max(min: Vector, max: Vector, ncells_axis: usize) -> Self {
    let rect = Rect::new_min_max(min, max);
    let ncells = vec![ncells_axis; rect.dim()];
    Self { rect, ncells }
  }
  pub fn new_unit(dim: Dim, ncells_axis: usize) -> Self {
    Self {
      rect: Rect::new_unit_cube(dim),
      ncells: vec![ncells_axis; dim],
    }
  }
  pub fn new_unit_scaled(dim: Dim, ncells_axis: usize, scale: f64) -> Self {
    Self {
      rect: Rect::new_scaled_cube(dim, scale),
      ncells: vec![ncells_axis; dim],
    }
  }
}
// getters
impl CartesianGrid {
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
  /// The cell count of each axis.
  pub fn ncells_per_axis(&self) -> &[usize] {
    &self.ncells
  }
  /// The vertex count of each axis: one more than its cells.
  pub fn nvertices_per_axis(&self) -> Vec<usize> {
    self.ncells.iter().map(|&n| n + 1).collect()
  }
  pub fn ncells(&self) -> usize {
    self.ncells.iter().product()
  }
  pub fn nvertices(&self) -> usize {
    self.nvertices_per_axis().iter().product()
  }
  pub fn vertex_cart_idx(&self, ivertex: usize) -> Vec<usize> {
    linear2cartesian_mixed(ivertex, &self.nvertices_per_axis())
  }
  pub fn vertex_pos(&self, ivertex: usize) -> Vector {
    let cart_idx = self.vertex_cart_idx(ivertex);
    let fractions = Vector::from_iterator(
      cart_idx.len(),
      cart_idx
        .iter()
        .zip(&self.ncells)
        .map(|(&c, &n)| c as f64 / n as f64),
    );
    fractions.component_mul(&self.side_lengths()) + self.min()
  }

  pub fn is_vertex_on_boundary(&self, vertex: usize) -> bool {
    self
      .vertex_cart_idx(vertex)
      .iter()
      .zip(&self.ncells)
      .any(|(&c, &n)| c == 0 || c == n)
  }

  pub fn boundary_vertices(&self) -> Vec<usize> {
    let nvertices = self.nvertices_per_axis();
    (0..self.nvertices())
      .filter(|&v| self.is_vertex_on_boundary(v))
      .map(|v| cartesian2linear_mixed(&self.vertex_cart_idx(v), &nvertices))
      .collect()
  }
}

impl CartesianGrid {
  /// A causally generic Minkowski spacetime box, axis $0$ the time direction in
  /// signature $(-, +, dots.c, +)$: the Kuhn triangulation of
  /// $[0, rho] times [0, 1]^(d-1)$ with the flat Minkowski metric as its
  /// ambient inner product, `ncells_axis` cells per side.
  ///
  /// The time axis is scaled by [`CAUSAL_TIME_SCALE`] so no mesh edge is
  /// lightlike -- the well-posedness condition of spacetime FEEC, a null edge
  /// degenerating the indefinite $L^2$ pairing on Whitney 1-forms. The returned
  /// coordinates carry the Minkowski ambient, so
  /// [`MeshCoords::to_edge_lengths_sq`] yields the signed Regge geometry
  /// directly; a Euclidean comparison view (to norm errors, which the indefinite
  /// pairing cannot) is [`MeshCoords::new`] on the same vertex matrix.
  pub fn minkowski(dim: Dim, ncells_axis: usize) -> (Complex, MeshCoords) {
    let mut max = Vector::from_element(dim, 1.0);
    if dim > 0 {
      max[0] = CAUSAL_TIME_SCALE;
    }
    let grid = Self::new_min_max(Vector::zeros(dim), max, ncells_axis);
    let (complex, coords) = grid.triangulate();
    let coords = MeshCoords::with_ambient(coords.into_matrix(), Gramian::minkowski(dim));
    debug_assert!(
      coords
        .to_edge_lengths_sq(&complex)
        .is_causally_generic(&complex),
      "a Minkowski box must have no lightlike edge"
    );
    (complex, coords)
  }

  pub fn triangulate(&self) -> (Complex, MeshCoords) {
    let (skeleton, coords) = self.triangulate_cells();
    let complex = Complex::from_cells(skeleton);
    (complex, coords)
  }
  pub fn triangulate_cells(&self) -> (Skeleton, MeshCoords) {
    let skeleton = self.cell_skeleton();
    let coords = self.vertex_coords();
    (skeleton, coords)
  }
  pub fn vertex_coords(&self) -> MeshCoords {
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
  pub fn cell_skeleton(&self) -> Skeleton {
    let dim = self.dim();
    let nvertices = self.nvertices_per_axis();
    let strides = mixed_strides(&nvertices);

    let mut simplices: Vec<Simplex> = Vec::with_capacity(factorial(dim) * self.ncells());
    for ibox in 0..self.ncells() {
      let box_cart = linear2cartesian_mixed(ibox, &self.ncells);
      let origin = cartesian2linear_mixed(&box_cart, &nvertices);

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
  use super::CartesianGrid;
  use crate::linalg::Matrix;

  #[test]
  fn unit_cube_mesh() {
    let (mesh, coords) = CartesianGrid::new_unit(3, 1).triangulate_cells();

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
    let (mesh, coords) = CartesianGrid::new_unit(2, 2).triangulate_cells();

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
