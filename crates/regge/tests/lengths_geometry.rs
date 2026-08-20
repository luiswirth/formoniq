//! [`CellGramians`] conformity: a Regge-conforming family induces edge
//! lengths, a family with a tangential jump is refused rather than resolved
//! arbitrarily.

use metric::Metric;
use multialgebra::Variance;
use multiindex::Dim;
use regge::lengths::geometry::CellGramians;
use regge::mesher::cartesian::CartesianGrid;

/// Tangential-tangential continuity is what makes the metric $->$ lengths
/// leg well defined at all, so a geometry that violates it has to be refused
/// rather than resolved in favor of whichever cell happens to be visited
/// last.
#[test]
fn non_conforming_cell_metrics_induce_no_edge_lengths() {
  for dim in 2..=3 {
    let dim = Dim::from(dim);
    let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
    let lengths = coords.to_edge_lengths_sq(&topology);

    let conforming = CellGramians::from_lengths(&topology, &lengths);
    assert!(conforming.is_regge_conforming(&topology));
    approx::assert_relative_eq!(
      conforming.to_edge_lengths_sq(&topology).vector(),
      lengths.vector(),
      epsilon = 1e-12
    );

    // Stretching one cell alone leaves its neighbors' lengths on the faces
    // they share untouched, which is exactly a tangential-tangential jump.
    let mut metrics: Vec<Metric> = topology
      .cells()
      .handle_iter()
      .map(|cell| lengths.cell_metric(cell))
      .collect();
    metrics[0] = Metric::new(Variance::Covariant, metrics[0].matrix() * 1.5);
    let broken = CellGramians::new(topology.dim(), metrics);

    assert!(!broken.is_regge_conforming(&topology));
    assert!(broken.try_to_edge_lengths_sq(&topology).is_none());
  }
}
