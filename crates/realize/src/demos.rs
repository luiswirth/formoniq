//! Meshes the reductions are demonstrated and tested on.
//!
//! Small, hand-written and flat, so a bake or a reduction can be checked
//! against a mesh whose incidence is small enough to state in the assertion
//! itself. They are fixtures with a shape worth reusing, not a mesh library:
//! anything generated belongs in `simplicial`'s meshers.

use simplicial::linalg::Matrix;
use simplicial::{
  geometry::coord::mesh::MeshCoords,
  topology::{complex::Complex, simplex::Simplex, skeleton::Skeleton},
};

/// The "triforce" teaching mesh: a central equilateral triangle with one
/// congruent triangle mirrored outward across each of its three edges, four
/// cells in all. The layout of the thesis figures' worked example of a global
/// shape function -- reused here rather than invented anew, so the
/// Whitney-basis study on it reproduces those figures.
/// Every interior edge is shared by exactly two cells and the one interior
/// vertex by three, enough to show a global shape function's multi-cell support
/// without the mesh itself being interesting.
///
/// Flat, and embedded in $RR^3$'s $z = 0$ plane: the viewer's one ambient space
/// is $RR^3$, and a planar mesh embeds as itself there.
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

  let columns: Vec<_> = positions.iter().map(|p| na::dvector![p[0], p[1]]).collect();
  let coords = MeshCoords::from(Matrix::from_columns(&columns)).embed_euclidean(3);
  let simplices = cells
    .into_iter()
    .map(|c| Simplex::from_word(c.to_vec()).1)
    .collect();
  (Complex::from_cells(Skeleton::new(simplices)), coords)
}
