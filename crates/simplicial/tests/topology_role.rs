//! Simplex roles: propositions about a simplex's dimension, checked once at
//! navigation and carried as proof, plus the dual-graph structure they give
//! (neighbors, facet adjacency, the ridge fan).

use simplicial::Dim;
use simplicial::mesher::grid::CartesianTopology;
use simplicial::topology::complex::Complex;
use simplicial::topology::role::{Cell, roles};

/// The role predicates, swept over all dimensions and grades: a role is
/// admitted exactly on its dimension, and roles coexist where their
/// dimensions coincide (the edge of a 1-complex is an edge and a cell),
/// propositions, not a partition.
#[test]
fn roles_are_admitted_exactly_on_their_dimension() {
  for top in (0..=4usize).map(Dim::from) {
    let complex = Complex::unit(top);
    for dim in top.range_inclusive() {
      for simplex in complex.skeleton(dim).handle_iter() {
        assert_eq!(simplex.as_role::<roles::Vertex>().is_some(), dim == 0);
        assert_eq!(simplex.as_role::<roles::Edge>().is_some(), dim == 1);
        assert_eq!(simplex.as_role::<roles::Cell>().is_some(), dim == top);
        assert_eq!(simplex.as_role::<roles::Facet>().is_some(), dim + 1 == top);
        assert_eq!(simplex.as_role::<roles::Ridge>().is_some(), dim + 2 == top);
      }
    }
  }
}

/// A face carries no cell proof: asserting one is a contract violation, and
/// the type, not a convention, is what says so.
#[test]
#[should_panic(expected = "is not a cell")]
fn a_face_is_not_a_cell() {
  let complex = Complex::unit(Dim::new(2));
  let edge = complex.skeleton(Dim::new(1)).handle_iter().next().unwrap();
  edge.role::<roles::Cell>();
}

/// Dual-graph adjacency is symmetric, and a cell has exactly one neighbor
/// per interior facet.
#[test]
fn neighboring_is_symmetric_and_facet_induced() {
  for dim in (1..=3usize).map(Dim::from) {
    let complex = CartesianTopology::cube(dim, 2).triangulate();
    for cell in complex.cells().handle_iter() {
      let interior = cell.facets().filter(|f| !f.is_boundary()).count();
      assert_eq!(cell.neighbors().count(), interior);
      for neighbor in cell.neighbors() {
        assert!(neighbor.neighbors().any(|back| back == cell));
      }
    }
  }
}

/// The generic accessor is total: `None` exactly where the complex has no
/// simplices of the role's dimension, never an underflow.
#[test]
fn role_skeletons_exist_exactly_where_their_dimension_does() {
  for top in (0..=4usize).map(Dim::from) {
    let complex = Complex::unit(top);
    assert!(complex.role_skeleton::<roles::Vertex>().is_some());
    assert_eq!(complex.role_skeleton::<roles::Edge>().is_some(), top >= 1);
    assert!(complex.role_skeleton::<roles::Cell>().is_some());
    assert_eq!(complex.role_skeleton::<roles::Facet>().is_some(), top >= 1);
    assert_eq!(complex.role_skeleton::<roles::Ridge>().is_some(), top >= 2);
  }
}

/// An edge's endpoints are its two vertices, in order, with their proofs.
#[test]
fn edge_endpoints_are_its_vertices() {
  for top in (1..=4usize).map(Dim::from) {
    let complex = Complex::unit(top);
    for edge in complex.edges().handle_iter() {
      let (a, b) = edge.endpoints();
      assert_eq!(vec![a.kidx(), b.kidx()], edge.simplex().vertices);
    }
  }
}

/// The fan of a ridge: every incident cell exactly once, consecutive cells
/// sharing a facet containing the ridge, closed exactly on interior ridges.
#[test]
fn ridge_fans_walk_the_hinge() {
  for dim in (2..=3usize).map(Dim::from) {
    let complex = CartesianTopology::cube(dim, 2).triangulate();
    for ridge in complex
      .role_skeleton::<roles::Ridge>()
      .unwrap()
      .handle_iter()
    {
      let fan = ridge.fan();
      let hinged = |a: Cell, b: Cell| {
        a.facets().any(|f| {
          ridge.simplex().is_subsimplex_of(f.simplex()) && b.facets().any(|g| g.idx() == f.idx())
        })
      };

      let mut fanned: Vec<_> = fan.iter().map(|cell| cell.idx()).collect();
      fanned.sort_unstable();
      assert!(fanned.windows(2).all(|w| w[0] != w[1]));
      let mut incident: Vec<_> = ridge.get().cells().map(|cell| cell.idx()).collect();
      incident.sort_unstable();
      assert_eq!(fanned, incident);

      for pair in fan.windows(2) {
        assert!(hinged(pair[0], pair[1]));
      }
      let interior = !ridge
        .get()
        .cofaces(dim - 1)
        .any(|f| f.role::<roles::Facet>().is_boundary());
      if interior {
        assert!(hinged(*fan.last().unwrap(), fan[0]));
      }
    }
  }
}

/// Every facet of the unit simplex is boundary: it bounds the one cell.
#[test]
fn unit_complex_is_all_boundary() {
  for top in (1..=4usize).map(Dim::from) {
    let complex = Complex::unit(top);
    for facet in complex.facets().handle_iter() {
      assert!(facet.is_boundary());
      let (cell, other) = facet.adjacent_cells();
      assert_eq!(cell.dim(), top);
      assert!(other.is_none());
    }
  }
}
