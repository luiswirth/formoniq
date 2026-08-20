//! [`MeshCoords`]: geometry at every grade matches the induced embedding
//! metric, an edge's Regge length is its endpoints' distance, and a
//! Minkowski ambient induces Lorentzian cell metrics.

use multiindex::Dim;
use regge::cell_volume;
use regge::coord::mesh::{MeshCoords, VertexRefExt};
use regge::coord::simplex::SimplexRefExt;
use regge::lengths::mesh::EdgeRefExt;
use regge::mesher::cartesian::CartesianGrid;

/// Geometry is defined on every simplex, not only the cells: the intrinsic
/// metric [`MeshLengthsSq::simplex_metric`] reads off a subsimplex's own edge
/// lengths equals the metric the embedding induces on that subsimplex (the
/// ambient inner product pulled back along its spanning vectors), at every
/// grade. The subsimplex generalization is exact, and well defined from the
/// edge data alone, no containing cell is consulted.
#[test]
fn simplex_metric_matches_induced_at_every_grade() {
  for dim in (1..=3usize).map(Dim::from) {
    let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
    let lengths = coords.to_edge_lengths_sq(&topology);
    for grade in (1..=dim.index()).map(Dim::from) {
      for simp in topology.skeleton(grade).handle_iter() {
        let from_lengths = lengths.simplex_metric(simp);
        let induced = coords
          .ambient()
          .pullback(&simp.coord_simplex(&coords).spanning_vectors());
        approx::assert_relative_eq!(from_lengths.matrix(), induced.matrix(), epsilon = 1e-12);
        // The volume accessor is total over the skeleton and agrees with the
        // metric's own volume factor.
        approx::assert_relative_eq!(
          lengths.simplex_volume(simp),
          cell_volume(&from_lengths),
          epsilon = 1e-12
        );
      }
    }
  }
}

/// The witness reads cohere across the layers: an edge's Regge squared
/// length is the squared distance of its endpoints' coordinates in the
/// inducing embedding.
#[test]
fn edge_length_is_endpoint_distance() {
  for dim in (1..=3usize).map(Dim::from) {
    let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
    let lengths_sq = coords.to_edge_lengths_sq(&topology);
    for edge in topology.edges().handle_iter() {
      let (vi, vj) = edge.endpoints();
      let displacement = vj.coord(&coords) - vi.coord(&coords);
      assert_eq!(edge.length_sq(&lengths_sq), displacement.norm_squared());
      assert_eq!(edge.length(&lengths_sq), displacement.norm());
    }
  }
}

/// A mesh embedded in Minkowski ambient space induces Lorentzian cell
/// metrics: on a coordinate-aligned mesh the induced metric of every cell
/// is congruent to $eta$ itself, so its signature is $(n - 1, 1)$ by
/// Sylvester's law of inertia, the same code path as the Euclidean
/// ambient, one signature among all.
#[test]
fn minkowski_ambient_induces_lorentzian_cell_metrics() {
  for dim in (1..=3usize).map(Dim::from) {
    let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
    let spacetime = MeshCoords::with_ambient(
      coords.matrix().clone(),
      metric::Metric::minkowski(dim.index()),
    );
    for cell in topology.cells().handle_iter() {
      let metric = spacetime.cell_metric(cell);
      assert_eq!(metric.signature(), (dim.index() - 1, 1));
      assert!(!metric.is_riemannian());
    }
  }
}

/// A Minkowski embedding realizes Lorentzian Regge data: the signed
/// squared edge lengths carry the causal character of every edge, and the
/// per-cell metric reconstructed from them is the same Lorentzian metric
/// the embedding induces, Regge calculus doing exactly what it was
/// invented for.
#[test]
fn lorentzian_ambient_realizes_lorentzian_regge_data() {
  use metric::CausalType;
  let (topology, coords) = CartesianGrid::new_unit(Dim::new(2), 1).triangulate();
  let mut matrix = coords.matrix().clone();
  matrix.row_mut(0).scale_mut(0.7);
  let spacetime = MeshCoords::with_ambient(matrix, metric::Metric::minkowski(2));
  let regge = spacetime.to_edge_lengths_sq(&topology);

  let mut seen = std::collections::HashSet::new();
  for edge in topology.edges().handle_iter() {
    seen.insert(edge.causal_type(&regge) as u8);
    match edge.causal_type(&regge) {
      CausalType::Timelike => assert!(edge.length_sq(&regge) < 0.0),
      CausalType::Null => assert_eq!(edge.length_sq(&regge), 0.0),
      CausalType::Spacelike => assert!(edge.length_sq(&regge) > 0.0),
    }
  }
  // The time-scaled mesh has both timelike and spacelike edges.
  assert!(seen.len() >= 2);

  for cell in topology.cells().handle_iter() {
    let from_regge = regge.cell_metric(cell);
    let from_coords = spacetime.cell_metric(cell);
    approx::assert_relative_eq!(from_regge.matrix(), from_coords.matrix(), epsilon = 1e-12);
    assert_eq!(from_regge.signature(), (1, 1));
  }
}
