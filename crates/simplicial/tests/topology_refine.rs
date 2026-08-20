//! Freudenthal refinement of a [`Complex`]: counts match the reference
//! pattern, conformity holds ($diff compose diff = 0$ on the refined
//! complex), and the totality at $R = 1$ and the point.

use simplicial::Dim;
use simplicial::linalg::CsrMatrix;
use simplicial::mesher::grid::CartesianTopology;
use simplicial::topology::complex::Complex;
use simplicial::topology::data::SkeletonData;

/// Refinement counts match the reference pattern, and the refined complex is
/// a valid manifold complex (its `from_cells` manifold check is the conformity
/// assertion). Vertices grow by the number of new lattice points per cell,
/// deduplicated across shared faces.
#[test]
fn refine_counts_and_conformity() {
  for dim in (1..=3usize).map(Dim::from) {
    let coarse = CartesianTopology::cube(dim, 2).triangulate();
    for r in 1..=3 {
      let sub = coarse.refine(r);
      let fine = sub.complex();

      // R^n children per coarse cell.
      assert_eq!(
        fine.nsimplices(dim),
        coarse.nsimplices(dim) * r.pow(dim.index() as u32)
      );
      // Coarse vertices keep their labels; refinement only adds vertices.
      assert!(sub.nvertices() >= coarse.vertices().len());
      assert_eq!(fine.vertices().len(), sub.nvertices());
      // The child provenance covers every refined cell.
      assert_eq!(sub.children().len(), fine.nsimplices(dim));
      // Boundary of the boundary vanishes: a valid chain complex was built.
      for k in (1..dim.index()).map(Dim::from) {
        let d0 = CsrMatrix::from(&fine.coboundary_operator(k - 1));
        let d1 = CsrMatrix::from(&fine.coboundary_operator(k));
        assert!((d1 * d0).values().iter().all(|&v| v == 0.0));
      }
    }
  }
}

/// A single triangle refines (red) into 4 triangles over 6 vertices; a single
/// tetrahedron into 8 over 10. The classical Bank/Bey counts.
#[test]
fn red_refinement_classical_counts() {
  let tri = Complex::unit(Dim::new(2)).refine(2);
  assert_eq!(tri.complex().nsimplices(Dim::new(2)), 4);
  assert_eq!(tri.nvertices(), 6);

  let tet = Complex::unit(Dim::new(3)).refine(2);
  assert_eq!(tet.complex().nsimplices(Dim::new(3)), 8);
  assert_eq!(tet.nvertices(), 10);
}

/// Totality: a 0-complex refines to itself, and $R = 1$ is the identity on
/// the cell count for every dimension.
#[test]
fn refine_degenerate_and_identity() {
  let points = Complex::unit(Dim::new(0)).refine(3);
  assert_eq!(points.complex().nsimplices(Dim::new(0)), 1);
  assert_eq!(points.nvertices(), 1);

  for dim in (0..=3usize).map(Dim::from) {
    let coarse = CartesianTopology::cube(dim, 2).triangulate();
    let identity = coarse.refine(1);
    assert_eq!(identity.complex().nsimplices(dim), coarse.nsimplices(dim));
    assert_eq!(identity.nvertices(), coarse.vertices().len());
  }
}
