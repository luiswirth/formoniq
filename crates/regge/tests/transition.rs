//! The atlas checked against an embedding.
//!
//! The transition map and its differential are metric-free facts about the
//! complex, stated in `simplicial`. Verifying them against coordinates needs an
//! embedding, so that check lives here: the claim is one crate down, the
//! witness one crate up.

use approx::assert_relative_eq;
use multiindex::Dim;
use regge::{coord::simplex::SimplexRefExt, mesher::cartesian::CartesianGrid};
use simplicial::atlas::ChartExt;

/// The differential of the transition is the change of frame between the two
/// charts, on the tangent space of the overlap.
///
/// Checked against an embedding, which both charts parametrize: a tangent
/// vector of the shared face, pushed into the ambient space through either
/// chart, is the same ambient vector. $A_(K') dif psi = A_K$ on $T sigma$.
#[test]
fn differential_is_the_change_of_frame() {
  for dim in (2..=3usize).map(Dim::from) {
    let (complex, coords) = CartesianGrid::new_unit(dim, 2).triangulate();

    for facet in complex.skeleton(dim - 1).handle_iter() {
      let cells: Vec<_> = facet.cells().collect();
      for (i, &source) in cells.iter().enumerate() {
        for &target in &cells[i + 1..] {
          let differential = source.transition_to(target).differential();

          // The tangent space of the shared facet, in the source chart.
          let positions = facet.simplex().relative_to(source.simplex());
          let tangents = simplicial::atlas::unit_face_spanning_vectors(dim, &positions);

          let source_frame = source.coord_simplex(&coords).linear_transform();
          let target_frame = target.coord_simplex(&coords).linear_transform();

          assert_relative_eq!(
            &target_frame * (&differential * &tangents),
            &source_frame * &tangents,
            epsilon = 1e-12
          );
        }
      }
    }
  }
}
