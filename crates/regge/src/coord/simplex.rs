//! The embedded specialization of [`SimplexCoords`]: the extrinsic bridges.
//!
//! The affine realization itself is generic over the coordinate space and lives
//! in [`simplicial::atlas::simplex_coords`]. What is added here is everything that
//! presupposes an *embedding* -- the `Ambient` instantiation `SimplexCoords`
//! (its default), whose vertices are points of $RR^N$:
//!
//! - constructing one from a mesh's [`MeshCoords`] and a topological
//!   [`Simplex`] ([`simplex_coords`], or [`SimplexRefExt::coord_simplex`] on a
//!   handle),
//! - the metric it *induces* ([`metric_tensor`](SimplexCoords::metric_tensor))
//!   and the Regge squared edge lengths it *realizes*
//!   ([`to_lengths_sq`](SimplexCoords::to_lengths_sq)).
//!
//! These are the two bridges down into the intrinsic layer, and they run
//! downward only: the metric layer never learns that coordinates exist
//! (invariant 2).

use super::mesh::MeshCoords;
use crate::metric::simplex::SimplexLengthsSq;
use simplicial::topology::{handle::SimplexRef, simplex::Simplex};

use coorder::Ambient;
use gramian::Gramian;
use simplicial::linalg::Matrix;

pub use simplicial::atlas::SimplexCoords;

/// The geometry an ambient realization induces.
///
/// `simplicial`'s [`SimplexCoords<Ambient>`] is the affine parametrization
/// $hat(K) -> RR^N$ and is metric-free. Reading a metric or edge lengths off it
/// is the bridge this crate is for, so it reaches down as an extension: an
/// embedding induces a metric, a metric induces no embedding.
pub trait SimplexCoordsExt {
  /// The metric a *Euclidean* ambient induces on this realization: the
  /// Gramian of the cell's spanning vectors. The general bridge is the
  /// pullback of the mesh's ambient inner product
  /// ([`MeshCoords::cell_metric`](crate::coord::mesh::MeshCoords::cell_metric)),
  /// of which this is the standard-signature case.
  fn metric_tensor(&self) -> Gramian;

  /// The Regge squared edge lengths this (Euclidean-ambient) coordinate
  /// realization has: the bridge from the extrinsic layer down into the
  /// intrinsic one.
  fn to_lengths_sq(&self) -> SimplexLengthsSq;
}

/// The affine parametrization a topological simplex has under an embedding:
/// its vertices' coordinates, as the columns.
///
/// A free function rather than a constructor on [`SimplexCoordsExt`]: it takes
/// the simplex and the coordinates on equal footing and has no receiver, so
/// there is no method syntax for a trait to carry.
pub fn simplex_coords(simp: &Simplex, coords: &MeshCoords) -> SimplexCoords {
  let mut vert_coords = Matrix::zeros(coords.dim().index(), simp.nvertices());
  for (i, v) in simp.iter().enumerate() {
    vert_coords.set_column(i, &coords.coord(v).view());
  }
  SimplexCoords::new(vert_coords)
}

impl SimplexCoordsExt for SimplexCoords<Ambient> {
  /// The metric a *Euclidean* ambient induces on this realization: the
  /// Gramian of the cell's spanning vectors. The general bridge is the
  /// pullback of the mesh's ambient inner product
  /// ([`MeshCoords::cell_metric`](crate::coord::mesh::MeshCoords::cell_metric)),
  /// of which this is the standard-signature case.
  fn metric_tensor(&self) -> Gramian {
    Gramian::from_euclidean_vectors(self.spanning_vectors())
  }

  /// The Regge squared edge lengths this (Euclidean-ambient) coordinate
  /// realization has: the bridge from the extrinsic layer down into the
  /// intrinsic one.
  fn to_lengths_sq(&self) -> SimplexLengthsSq {
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
    simplex_coords(self.simplex(), coords)
  }
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::metric::simplex::SimplexLengthsSq;
  use multiindex::Dim;
  use simplicial::atlas::unit_vertices;
  use simplicial::linalg::Vector;

  use approx::assert_relative_eq;

  /// The standard coordinate simplex realizes the standard squared edge
  /// lengths: the two descriptions of the reference cell agree, extrinsic
  /// and intrinsic.
  #[test]
  fn unit_coords_realize_unit_lengths() {
    for dim in (0..=4usize).map(Dim::from) {
      let coords: SimplexCoords = SimplexCoords::new(unit_vertices(dim));
      let lengths_sq = coords.to_lengths_sq();
      assert_relative_eq!(lengths_sq.vector(), SimplexLengthsSq::unit(dim).vector());
      assert_relative_eq!(coords.vol(), lengths_sq.vol());
    }
  }

  /// The induced metric of the reference cell is the identity: its spanning
  /// vectors are the orthonormal standard basis.
  #[test]
  fn unit_metric_is_identity() {
    for dim in (1..=4usize).map(Dim::from) {
      let coords: SimplexCoords = SimplexCoords::new(unit_vertices(dim));
      assert_relative_eq!(
        coords.metric_tensor().matrix(),
        &Matrix::identity(dim.index(), dim.index())
      );
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
