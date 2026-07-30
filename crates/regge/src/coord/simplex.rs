//! The affine parametrization a topological simplex has under an embedding.
//!
//! The realization itself is generic over the coordinate space and lives in
//! [`simplicial::atlas::simplex_coords`]. What is added here is the one thing
//! that presupposes an embedding: reading the `Ambient` instantiation
//! `SimplexCoords` (its default) off a mesh's [`MeshCoords`] and a topological
//! [`Simplex`].
//!
//! The geometry such a realization induces is not read here, and cannot be. A
//! `SimplexCoords` carries vertex positions and no inner product on the space
//! they live in, so a metric taken off it alone could only assume the Euclidean
//! one, and would be silently wrong on the Minkowski ambient this crate exists
//! to support. The ambient lives on the mesh, so the bridges do too:
//! [`MeshCoords::simplex_metric`] and [`MeshCoords::to_edge_lengths_sq`], each
//! a pullback of the ambient inner product. They run downward only: the metric
//! layer never learns that coordinates exist (invariant 2).

use super::mesh::MeshCoords;
use simplicial::topology::{handle::SimplexRef, simplex::Simplex};

use simplicial::linalg::Matrix;

pub use simplicial::atlas::SimplexCoords;

/// The affine parametrization a topological simplex has under an embedding:
/// its vertices' coordinates, as the columns.
///
/// A free function: it takes the simplex and the coordinates on equal footing
/// and has no receiver, so there is no method syntax for a trait to carry.
pub fn simplex_coords(simp: &Simplex, coords: &MeshCoords) -> SimplexCoords {
  let mut vert_coords = Matrix::zeros(coords.dim().index(), simp.nvertices());
  for (i, v) in simp.iter().enumerate() {
    vert_coords.set_column(i, &coords.coord(v).view());
  }
  SimplexCoords::new(vert_coords)
}

/// The affine parametrization of a cell, given an embedding: an `exterior`-free
/// coordinate construction on a topology handle, which is how invariant 1 is
/// upheld below crate granularity.
pub trait SimplexRefExt {
  fn coord_simplex(&self, coords: &MeshCoords) -> SimplexCoords;
}
impl SimplexRefExt for SimplexRef<'_> {
  fn coord_simplex(&self, coords: &MeshCoords) -> SimplexCoords {
    simplex_coords(self.simplex(), coords)
  }
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::lengths::mesh::MeshLengthsSq;
  use multiindex::Dim;
  use simplicial::linalg::Vector;
  use simplicial::topology::complex::Complex;

  use approx::assert_relative_eq;

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
}
