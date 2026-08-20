//! [`PointLocator`]: the BVH locator agrees with a brute-force scan on
//! every query, at any scale.

use multiindex::Dim;
use regge::coord::Coord;
use regge::coord::locate::PointLocator;
use regge::coord::simplex::SimplexRefExt;
use regge::mesher::cartesian::CartesianGrid;

/// The BVH locator agrees with the brute-force linear scan on every query,
/// and reports points outside the mesh as such.
///
/// Swept over the scale of the mesh as well as its dimension: which cell
/// holds a point is invariant under scaling the geometry and the query
/// together, so the tolerances the locator applies must be too.
#[test]
fn locator_matches_brute_force() {
  for dim in (1..=3usize).map(Dim::from) {
    for scale in [1e-4, 1.0, 1e4] {
      let (topology, mut coords) = CartesianGrid::new_unit(dim, 3).triangulate();
      *coords.matrix_mut() *= scale;
      let locator = PointLocator::new(&topology, &coords);

      // A grid of probe points covering the unit cube and a margin outside it.
      // A distinct per-axis phase keeps the points off cell faces and off the
      // triangulation diagonals, where the strict and tolerant tests would
      // legitimately disagree on measure-zero boundary sets.
      let samples: usize = 5;
      let phase = [0.041, 0.113, 0.237];
      let probe =
        |i: usize, d: usize| scale * (-0.2 + 1.4 * (i as f64 + phase[d]) / samples as f64);
      for flat in 0..samples.pow(dim.index() as u32) {
        let x = Coord::from_iterator(
          dim.index(),
          (0..dim.index()).map(|d| probe(flat / samples.pow(d as u32) % samples, d)),
        );

        let brute = coords.find_cell_containing(&topology, x.as_view());
        let found = locator.locate(&x);

        match (brute, &found) {
          (Some(_), Some(loc)) => {
            // The located cell must actually contain x, and its barycentric
            // coordinates must reconstruct the point.
            let simp = loc.chart(&topology).coord_simplex(&coords);
            assert!(simp.is_global_inside(x.as_view()), "dim={dim} x={x:?}");
            let reconstructed = simp.bary2global(loc.bary());
            assert!((&reconstructed - &x).norm() < 1e-9 * scale);
          }
          (None, None) => {}
          (a, b) => panic!(
            "disagreement at x={x:?}: brute={} bvh={}",
            a.is_some(),
            b.is_some()
          ),
        }
      }
    }
  }
}
