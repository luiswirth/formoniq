//! The cocycle law of the atlas: the transitions between charts compose.

use simplicial::Dim;
use simplicial::atlas::{ChartExt, barycenter_bary};
use simplicial::mesher::grid::CartesianTopology;

/// $psi_(K'' K') compose psi_(K' K) = psi_(K'' K)$: the cocycle condition, on
/// the triple overlap where all three charts see the point.
///
/// This is the coherence law of an atlas, the statement that the charts
/// describe one manifold and not three.
#[test]
fn transition_cocycle() {
  for dim in (2..=3usize).map(Dim::from) {
    let complex = CartesianTopology::cube(dim, 2).triangulate();

    // A vertex of the mesh lies in the overlap of every cell around it.
    for vertex in complex.vertices().handle_iter() {
      let cells: Vec<_> = vertex.cells().collect();
      for &first in &cells {
        let positions = vertex.simplex().relative_to(first.simplex());
        let point = first.point_on_face(&positions, &barycenter_bary(Dim::new(0)));

        for &second in &cells {
          for &third in &cells {
            let direct = first.transition_to(third).apply(&point).unwrap();
            let composed = first
              .transition_to(second)
              .apply(&point)
              .and_then(|mid| second.transition_to(third).apply(&mid))
              .unwrap();
            assert_eq!(direct, composed);
          }
        }
      }
    }
  }
}
