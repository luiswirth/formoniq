//! The two rungs of the manifold condition: pseudomanifold (every facet in
//! exactly two cells) and homology manifold (every vertex link a sphere).

use simplicial::mesher::grid::CartesianTopology;
use simplicial::topology::complex::Complex;
use simplicial::topology::simplex::Simplex;
use simplicial::topology::skeleton::Skeleton;

/// A triangulated box is a manifold with boundary at every dimension it can
/// be built in: interior vertices have spherical links, boundary ones have
/// acyclic links, and both are accepted.
#[test]
fn a_triangulated_box_is_a_manifold() {
  for dim in 1..=3 {
    let complex = CartesianTopology::cube(dim, 2).triangulate();
    assert!(complex.is_pseudomanifold(), "dim {dim}");
    assert!(complex.is_homology_manifold(), "dim {dim}");
  }
}

/// The boundary of a simplex is a closed manifold, a sphere: every link is
/// spherical, none acyclic.
#[test]
fn the_boundary_of_a_simplex_is_a_manifold() {
  for dim in 2..=4 {
    let complex = Complex::unit(dim)
      .boundary_complex()
      .expect("a simplex has a boundary")
      .complex()
      .clone();
    assert!(complex.is_pseudomanifold(), "dim {dim}");
    assert!(complex.is_homology_manifold(), "dim {dim}");
  }
}

/// Two triangles meeting at a single vertex: every facet lies in exactly one
/// cell, so it passes the pseudomanifold rung, and the link of the shared
/// vertex is two disjoint arcs rather than one, so it fails the homology one.
///
/// The witness that the two rungs are genuinely different conditions, and
/// that the cheap one is not the manifold condition.
#[test]
fn a_pinch_point_is_a_pseudomanifold_but_not_a_manifold() {
  let complex = Complex::from_cells(Skeleton::new(vec![
    Simplex::new(vec![0, 1, 2]),
    Simplex::new(vec![2, 3, 4]),
  ]));
  assert!(complex.is_pseudomanifold());
  assert!(!complex.is_homology_manifold());

  let link = complex.vertex_link(2).expect("a 2-complex has links");
  assert_eq!(link.complex().betti_numbers()[0], 2, "two disjoint arcs");
}
