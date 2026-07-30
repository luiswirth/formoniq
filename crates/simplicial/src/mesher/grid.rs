//! The combinatorial grid: a box of cells, triangulated.
//!
//! Only the per-axis cell counts, so no coordinates and no metric. The Kuhn
//! (Freudenthal) triangulation is an affine construction, and the vertex order
//! it emits is what makes uniform refinement compose, so it is combinatorics
//! and belongs with the topology. `regge::mesher::cartesian` places the
//! vertices in space.

use multiindex::{Combination, Dim, Permutation, Radix, factorial};

use crate::topology::{complex::Complex, simplex::Simplex, skeleton::Skeleton};

/// A box of `ncells` cells per axis, as combinatorics: the vertex grid and the
/// simplices of its Kuhn triangulation, with no notion of where any vertex is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CartesianTopology {
  cells: Radix,
}

impl CartesianTopology {
  /// From the number of cells along each axis.
  pub fn new(ncells: Vec<usize>) -> Self {
    assert!(
      ncells.iter().all(|&n| n > 0),
      "a grid has at least one cell per axis"
    );
    Self {
      cells: Radix::new(ncells),
    }
  }

  /// A cube of `ncells_axis` cells along each of `dim` axes.
  pub fn cube(dim: impl Into<Dim>, ncells_axis: usize) -> Self {
    Self::new(vec![ncells_axis; dim.into().index()])
  }

  pub fn dim(&self) -> Dim {
    self.cells.naxes().into()
  }

  /// The shape of the cell grid: the cell count of each axis.
  pub fn cell_shape(&self) -> &Radix {
    &self.cells
  }
  /// The shape of the vertex grid: one more vertex than cells along each axis.
  pub fn vertex_shape(&self) -> Radix {
    self.cells.radices().iter().map(|&n| n + 1).collect()
  }
  pub fn ncells(&self) -> usize {
    self.cells.count()
  }
  pub fn nvertices(&self) -> usize {
    self.vertex_shape().count()
  }
  pub fn vertex_cart_idx(&self, ivertex: usize) -> Vec<usize> {
    self.vertex_shape().delinearize(ivertex).to_vec()
  }
  pub fn is_vertex_on_boundary(&self, vertex: usize) -> bool {
    self
      .vertex_cart_idx(vertex)
      .iter()
      .zip(self.cells.radices())
      .any(|(&c, &n)| c == 0 || c == n)
  }
  pub fn boundary_vertices(&self) -> Vec<usize> {
    (0..self.nvertices())
      .filter(|&v| self.is_vertex_on_boundary(v))
      .collect()
  }
  /// Kuhn (Freudenthal) triangulation.
  ///
  /// The corners of a cube are the subsets of the axes (radix-2 cartesian
  /// indices are `Combination` bitsets), and each of the $d!$ simplices of a
  /// cube is a maximal chain
  /// $emptyset subset {a_1} subset {a_1, a_2} subset dots.c$
  /// in this subset lattice, one per permutation of the axes.
  pub fn cell_skeleton(&self) -> Skeleton {
    let dim = self.dim().index();
    let vertices_shape = self.vertex_shape();

    let mut simplices: Vec<Simplex> = Vec::with_capacity(factorial(dim) * self.ncells());
    for ibox in 0..self.ncells() {
      let origin = vertices_shape.linearize(&self.cells.delinearize(ibox));

      for axes in Permutation::all(dim) {
        let chain = axes.iter().scan(Combination::empty(), |corner, axis| {
          *corner = corner.inserted(axis);
          Some(*corner)
        });
        let vertices = std::iter::once(origin)
          .chain(chain.map(|corner| origin + vertices_shape.corner_offset(corner)))
          .collect();
        simplices.push(Simplex::new(vertices));
      }
    }

    Skeleton::new(simplices)
  }

  /// The Kuhn triangulation as a complex.
  pub fn triangulate(&self) -> Complex {
    Complex::from_cells(self.cell_skeleton())
  }
}

#[cfg(test)]
mod test {
  use super::CartesianTopology;

  /// The Kuhn triangulation of the unit cube has $d!$ cells per box, in colex
  /// order, and the cells are the maximal chains of the subset lattice.
  ///
  /// Stated on the combinatorics alone, with no coordinates in sight, which is
  /// the point of the split.
  #[test]
  fn the_kuhn_cells_are_the_maximal_chains() {
    let grid = CartesianTopology::cube(3, 1);
    assert_eq!(grid.ncells(), 1);
    assert_eq!(grid.nvertices(), 8);

    let cells: Vec<Vec<usize>> = grid
      .cell_skeleton()
      .iter()
      .map(|s| s.vertices.clone())
      .collect();
    assert_eq!(
      cells,
      vec![
        vec![0, 1, 3, 7],
        vec![0, 2, 3, 7],
        vec![0, 1, 5, 7],
        vec![0, 4, 5, 7],
        vec![0, 2, 6, 7],
        vec![0, 4, 6, 7],
      ]
    );
  }

  /// Every dimension and refinement gives $d!$ cells per box and the expected
  /// vertex count, so the counting is total rather than checked at one size.
  #[test]
  fn the_counts_hold_at_every_size() {
    for dim in 1..=4 {
      for ncells_axis in 1..=3 {
        let grid = CartesianTopology::cube(dim, ncells_axis);
        assert_eq!(grid.ncells(), ncells_axis.pow(dim as u32));
        assert_eq!(grid.nvertices(), (ncells_axis + 1).pow(dim as u32));
        assert_eq!(
          grid.cell_skeleton().len(),
          multiindex::factorial(dim) * grid.ncells()
        );
      }
    }
  }
}
