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
