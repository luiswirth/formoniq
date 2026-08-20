//! [`CartesianGrid`]: the frozen coordinate and cell layout of the Kuhn
//! triangulation, checked explicitly at two sizes.

use multiindex::Dim;
use regge::mesher::cartesian::CartesianGrid;
use simplicial::linalg::Matrix;

#[test]
fn unit_cube_mesh() {
  let (mesh, coords) = CartesianGrid::new_unit(Dim::new(3), 1).triangulate_cells();

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
  let (mesh, coords) = CartesianGrid::new_unit(Dim::new(2), 2).triangulate_cells();

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
