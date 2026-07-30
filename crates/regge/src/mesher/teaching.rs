//! Small hand-written meshes, stated cell by cell.
//!
//! A generated mesh is a family and a hand-written one is a shape: these are
//! chosen for the figure they make, and they are small enough that a claim
//! about their incidence can be written into an assertion.

use crate::coord::mesh::MeshCoords;
use simplicial::linalg::Matrix;
use simplicial::topology::{complex::Complex, simplex::Simplex, skeleton::Skeleton};

/// The "triforce" mesh: a central equilateral triangle with one congruent
/// triangle mirrored outward across each of its three edges, four cells in
/// all.
///
/// Every interior edge is shared by exactly two cells and the one interior
/// vertex by three, which is the smallest mesh on which a basis function
/// supported on several cells has anything to show.
///
/// Flat, and embedded in $RR^3$'s $z = 0$ plane, a planar mesh embedding as
/// itself there.
pub fn triforce() -> (Complex, MeshCoords) {
  let sqrt3_2 = 3f64.sqrt() / 2.0;
  #[rustfmt::skip]
  let positions: [[f64; 2]; 6] = [
    [ 0.0, 0.0],
    [ 1.0, 0.0],
    [ 0.5, sqrt3_2],
    [-0.5, sqrt3_2],
    [ 1.5, sqrt3_2],
    [ 0.5, -sqrt3_2],
  ];
  let cells: [[usize; 3]; 4] = [[0, 1, 2], [0, 2, 3], [1, 4, 2], [0, 1, 5]];

  let columns: Vec<_> = positions
    .iter()
    .map(|p| simplicial::linalg::Vector::from_vec(vec![p[0], p[1]]))
    .collect();
  let coords = MeshCoords::from(Matrix::from_columns(&columns)).embed_euclidean(3);
  let simplices = cells
    .into_iter()
    .map(|c| Simplex::from_word(c.to_vec()).1)
    .collect();
  (Complex::from_cells(Skeleton::new(simplices)), coords)
}
