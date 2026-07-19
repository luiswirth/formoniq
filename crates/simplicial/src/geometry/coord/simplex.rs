//! The embedded specialization of [`SimplexCoords`]: the extrinsic bridges.
//!
//! The affine realization itself is generic over the coordinate space and lives
//! in [`crate::atlas::simplex_coords`]. What is added here is everything that
//! presupposes an *embedding* -- the `Ambient` instantiation `SimplexCoords`
//! (its default), whose vertices are points of $RR^N$:
//!
//! - constructing one from a mesh's [`MeshCoords`] and a topological
//!   [`Simplex`] ([`SimplexCoords::from_simplex_and_coords`], the
//!   [`SimplexRefExt`] handle method),
//! - the metric it *induces* ([`metric_tensor`](SimplexCoords::metric_tensor))
//!   and the Regge squared edge lengths it *realizes*
//!   ([`to_lengths_sq`](SimplexCoords::to_lengths_sq)).
//!
//! These are the two bridges down into the intrinsic layer, and they run
//! downward only: the metric layer never learns that coordinates exist
//! (invariant 2).

use super::mesh::MeshCoords;
use crate::{
  geometry::metric::simplex::SimplexLengthsSq,
  topology::{handle::SimplexRef, simplex::Simplex},
};

use crate::linalg::Matrix;
use coorder::Ambient;
use gramian::Gramian;

pub use crate::atlas::SimplexCoords;

impl SimplexCoords<Ambient> {
  pub fn from_simplex_and_coords(simp: &Simplex, coords: &MeshCoords) -> SimplexCoords {
    let mut vert_coords = Matrix::zeros(coords.dim(), simp.nvertices());
    for (i, v) in simp.iter().enumerate() {
      vert_coords.set_column(i, &coords.coord(v).view());
    }
    SimplexCoords::new(vert_coords)
  }

  /// The metric a *Euclidean* ambient induces on this realization: the
  /// Gramian of the cell's spanning vectors. The general bridge is the
  /// pullback of the mesh's ambient inner product
  /// ([`MeshCoords::cell_metric`](crate::geometry::coord::mesh::MeshCoords::cell_metric)),
  /// of which this is the standard-signature case.
  pub fn metric_tensor(&self) -> Gramian {
    Gramian::from_euclidean_vectors(self.spanning_vectors())
  }

  /// The Regge squared edge lengths this (Euclidean-ambient) coordinate
  /// realization has: the bridge from the extrinsic layer down into the
  /// intrinsic one.
  pub fn to_lengths_sq(&self) -> SimplexLengthsSq {
    let lengths_sq: Vec<f64> = self.edges().map(|e| e.vol().powi(2)).collect();
    // SAFETY: Squared lengths stem from a realization already.
    SimplexLengthsSq::new_unchecked(lengths_sq.into(), self.dim_intrinsic())
  }
}

/// The affine parametrization of a cell, given an embedding: an `exterior`-free
/// coordinate construction on a topology handle, which is how invariant 1 is
/// upheld below crate granularity.
pub trait SimplexRefExt {
  fn coord_simplex(&self, coords: &MeshCoords) -> SimplexCoords;
}
impl SimplexRefExt for SimplexRef<'_> {
  fn coord_simplex(&self, coords: &MeshCoords) -> SimplexCoords {
    SimplexCoords::from_simplex_and_coords(self.simplex(), coords)
  }
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::atlas::ref_vertices;
  use crate::geometry::metric::simplex::SimplexLengthsSq;
  use crate::linalg::Vector;

  use approx::assert_relative_eq;

  /// The standard coordinate simplex realizes the standard squared edge
  /// lengths: the two descriptions of the reference cell agree, extrinsic
  /// and intrinsic.
  #[test]
  fn ref_coords_realize_ref_lengths() {
    for dim in 0..=4 {
      let coords: SimplexCoords = SimplexCoords::new(ref_vertices(dim));
      let lengths_sq = coords.to_lengths_sq();
      assert_relative_eq!(
        lengths_sq.vector(),
        SimplexLengthsSq::standard(dim).vector()
      );
      assert_relative_eq!(coords.vol(), lengths_sq.vol());
    }
  }

  /// The induced metric of the reference cell is the identity: its spanning
  /// vectors are the orthonormal standard basis.
  #[test]
  fn ref_metric_is_identity() {
    for dim in 1..=4 {
      let coords: SimplexCoords = SimplexCoords::new(ref_vertices(dim));
      assert_relative_eq!(coords.metric_tensor().matrix(), &Matrix::identity(dim, dim));
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
