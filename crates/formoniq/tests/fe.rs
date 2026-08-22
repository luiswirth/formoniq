//! The $L^2$ projection onto the Whitney space: $P_h compose W = id$, so a
//! discrete form is its own best approximation.

use approx::assert_relative_eq;
use derham::Cochain;
use derham::interpolate::interpolant::WhitneyInterpolant;
use formoniq::fe::l2_projection;
use formoniq::whitney_complex::WhitneyComplex;
use regge::mesher::cartesian::CartesianGrid;
use simplicial::atlas::SimplexQuadRule;
use simplicial::{Dim, linalg::Vector};

/// $P_h compose W = id$: the $L^2$ projection is the identity on the Whitney
/// space, since a discrete form is its own best approximation.
///
/// The sharpest available check that the Hodge mass matrix and the source
/// load are the same bilinear form seen from two sides.
#[test]
fn l2_projection_reproduces_whitney_forms() {
  for dim in (1..=3).map(Dim::from) {
    let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
    let lengths = coords.to_edge_lengths_sq(&topology);
    let whitney = WhitneyComplex::new(&topology, &lengths);

    for grade in dim.range_inclusive() {
      let ndofs = topology.nsimplices(grade);
      let cochain = Cochain::new(
        grade,
        Vector::from_iterator(ndofs, (0..ndofs).map(|i| ((i % 7) as f64) - 3.0)),
      );

      let field = WhitneyInterpolant::new(cochain.clone(), &topology);
      let qr = SimplexQuadRule::degree(dim, 3);
      let projected = l2_projection(&field, whitney, Some(qr));

      assert_relative_eq!(projected.coeffs(), cochain.coeffs(), epsilon = 1e-9);
    }
  }
}
