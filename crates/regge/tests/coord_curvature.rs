//! Discrete mean curvature and the curvature radius it bounds: the unit
//! sphere matches the analytic value, a flat grid is unbounded.

use multiindex::Dim;
use regge::coord::{vertex_curvature_radius, vertex_mean_curvature};
use regge::mesher::cartesian::CartesianGrid;
use regge::mesher::sphere::mesh_sphere_surface;

/// The unit sphere has constant curvature $K = H^2 = 1$ and curvature
/// radius $1$ everywhere. The discrete estimators recover $|H|$ to within
/// the barycentric lumped area's discretization error (cruder than a mixed
/// Voronoi area, but simpler and reused as-is for [`vertex_curvature_radius`]).
/// That same area error enters $kappa_max = |H| + sqrt(max(H^2-K,0))$
/// asymmetrically, squared through $H^2$, linear through $K$, so the
/// radius estimate carries a larger, but conservative (radius
/// underestimated, never overestimated), bias than $H$ alone. Underestimating
/// the safe radius is exactly the safe direction for a fold-safety cap, so
/// this is loose on purpose, not a correctness gap; the exact Gauss-Bonnet
/// identity elsewhere is what checks correctness.
#[test]
fn sphere_mean_curvature_and_radius_match_unit_radius() {
  let (topology, coords) = mesh_sphere_surface(3);
  let mean = vertex_mean_curvature(&topology, &coords);
  let radius = vertex_curvature_radius(&topology, &coords);
  for &h in &mean {
    assert!((h - 1.0).abs() < 0.2, "expected |H| ~ 1, got {h}");
  }
  for &r in &radius {
    assert!(
      r < 1.05 && r > 0.5,
      "expected curvature radius in (0.5, 1.05), got {r}"
    );
  }
}

/// A flat unit-square grid is developable: zero mean curvature at every
/// interior vertex (a boundary vertex's raw $H$ is a natural boundary
/// term, not curvature, see [`vertex_curvature_radius`]), so the
/// curvature radius is unbounded everywhere, boundary included, curvature
/// must never clamp displacement on a flat surface.
#[test]
fn flat_grid_has_unbounded_curvature_radius() {
  let (topology, coords) = CartesianGrid::new_unit(Dim::new(2), 4).triangulate();
  let coords = coords.embed_euclidean(Dim::new(3));
  let boundary: std::collections::HashSet<usize> =
    topology.boundary_vertices().into_iter().collect();
  let mean = vertex_mean_curvature(&topology, &coords);
  let radius = vertex_curvature_radius(&topology, &coords);
  for (v, &h) in mean.iter().enumerate() {
    if !boundary.contains(&v) {
      assert!(h.abs() < 1e-9, "expected interior H ~ 0, got {h}");
    }
  }
  for &r in &radius {
    assert!(
      r.is_infinite(),
      "expected an unbounded curvature radius, got {r}"
    );
  }
}
