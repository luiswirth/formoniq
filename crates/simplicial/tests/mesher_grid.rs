//! The Kuhn triangulation of a cartesian grid: purely combinatorial, no
//! coordinates involved.

use simplicial::mesher::grid::CartesianTopology;

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
