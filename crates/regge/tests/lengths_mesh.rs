//! [`MeshLengthsSq`]: the trace law a face's metric obeys against any
//! containing cell's, which is what makes the edge lengths and not the
//! per-cell Gramians the primitive.

use multiindex::Dim;
use regge::mesher::cartesian::CartesianGrid;

/// The trace law [`MeshLengthsSq::simplex_metric`] rests on: a face's
/// intrinsic metric is the pullback of any containing cell's along the
/// inclusion of tangent spaces, $g_sigma = B^T g_K B$.
///
/// This is the tangential-tangential trace of the Regge field, and it is
/// what makes the edge lengths rather than the per-cell Gramians the
/// primitive: the trace exists at every grade, so a face carries a metric
/// with no containing cell consulted. Swept over every dimension, every
/// grade and every face, the top grade included, where the inclusion is the
/// identity.
#[test]
fn subsimplex_metric_is_restriction_of_cell_metric() {
  for dim in 1..=4 {
    let dim = Dim::new(dim);
    let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
    let lengths = coords.to_edge_lengths_sq(&topology);

    for cell in topology.cells().handle_iter() {
      let cell_simplex = cell.get().simplex().clone();
      let cell_metric = lengths.cell_metric(cell);

      for grade in 1..=dim.index() {
        for face in cell.get().faces(grade) {
          let positions = face.simplex().relative_to(&cell_simplex);

          // Column $a$ is the face's tangent vector $u_(a+1) - u_0$ read in
          // the cell's basis $e_i = v_(i+1) - v_0$, where the apex $v_0$
          // contributes nothing because $e_(-1) = 0$.
          let mut inclusion = simplicial::linalg::Matrix::zeros(dim.index(), grade);
          let apex = positions.index_at(0);
          for a in 0..grade {
            let head = positions.index_at(a + 1);
            if head > 0 {
              inclusion[(head - 1, a)] += 1.0;
            }
            if apex > 0 {
              inclusion[(apex - 1, a)] -= 1.0;
            }
          }

          let restricted = cell_metric.pullback(&inclusion);
          let intrinsic = lengths.simplex_metric(face);
          approx::assert_relative_eq!(intrinsic.matrix(), restricted.matrix(), epsilon = 1e-12);
        }
      }
    }
  }
}
