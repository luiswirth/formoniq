//! A cell's extrinsic realization: the reference cell reproduces the
//! reference geometry, and an embedded cell's volume is the intrinsic one.

use approx::assert_relative_eq;
use multiindex::Dim;
use regge::coord::mesh::MeshCoords;
use regge::coord::simplex::SimplexCoords;
use regge::lengths::mesh::MeshLengthsSq;
use simplicial::linalg::{Matrix, Vector};
use simplicial::topology::complex::Complex;

/// The reference cell's two descriptions agree: the squared edge lengths its
/// coordinate realization induces are the reference ones, and its induced
/// metric is the identity, its spanning vectors being the orthonormal
/// standard basis.
///
/// Read through [`MeshCoords`], which is where the ambient inner product
/// lives: the realization on its own has vertex positions and no way to
/// measure them.
#[test]
fn the_reference_cell_realizes_the_reference_geometry() {
  for dim in (0..=4usize).map(Dim::from) {
    let topology = Complex::unit(dim);
    let coords = MeshCoords::unit(dim);

    assert_relative_eq!(
      coords.to_edge_lengths_sq(&topology).vector(),
      MeshLengthsSq::unit(dim).vector()
    );
    for cell in topology.cells().handle_iter() {
      assert_relative_eq!(
        coords.cell_metric(cell).matrix(),
        &Matrix::identity(dim.index(), dim.index())
      );
    }
  }
}

/// A lower-dimensional cell embedded in a higher-dimensional ambient space
/// has its intrinsic volume, read through the Gram (non-square) branch of
/// [`SimplexCoords::vol`]: a unit right triangle placed into $RR^3$ keeps area
/// $1 \/ 2$.
#[test]
fn embedded_volume_is_intrinsic() {
  let coords: SimplexCoords = SimplexCoords::new(Matrix::from_columns(&[
    Vector::from_column_slice(&[0.0, 0.0, 0.0]),
    Vector::from_column_slice(&[1.0, 0.0, 0.0]),
    Vector::from_column_slice(&[0.0, 1.0, 0.0]),
  ]));
  assert!(!coords.is_same_dim());
  assert_relative_eq!(coords.vol(), 0.5, epsilon = 1e-12);
}
