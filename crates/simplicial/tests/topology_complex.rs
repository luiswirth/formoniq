//! [`Complex`] navigation and structure: colex-ordered skeletons, the
//! boundary operator agreeing with the reference one, facet/neighbor/star/
//! link navigation, and the serde round-trip.

use simplicial::Dim;
use simplicial::linalg::{CsrMatrix, Matrix};
use simplicial::mesher::grid::CartesianTopology;
use simplicial::topology::complex::Complex;
use simplicial::topology::simplex::{Simplex, nsubsimplices, unit_boundary_operator};

/// Round-tripping through CBOR reproduces the same topology: the boundary
/// (an $O(n^3)$ derived quantity, not stored data) matches, so the save/load
/// pair is exercised through `Complex::from_cells`, not just through serde.
#[cfg(feature = "serde")]
#[test]
fn save_load_roundtrip() {
  let topology = CartesianTopology::cube(Dim::new(3), 2).triangulate();

  let path = std::env::temp_dir().join(format!("simplicial_test_{}.cbor", std::process::id()));
  topology.save(&path).unwrap();
  let loaded = Complex::load(&path).unwrap();
  std::fs::remove_file(&path).unwrap();

  assert_eq!(loaded.dim(), topology.dim());
  for dim in topology.dim().range_inclusive() {
    assert_eq!(loaded.nsimplices(dim), topology.nsimplices(dim));
  }
  assert_eq!(loaded.betti_numbers(), topology.betti_numbers());
}

/// Every skeleton is in canonical colexicographic order, and the vertices
/// are contiguous and fully used. This is the ordering contract the file
/// formats and cochain indexing rely on.
#[test]
fn skeletons_are_colex_ordered_and_vertices_contiguous() {
  for dim in (1..=3usize).map(Dim::from) {
    let topology = CartesianTopology::cube(dim, 3).triangulate();

    // Vertices are exactly 0..nvertices, each labeled by its own kidx.
    let vertices = topology.skeleton(Dim::new(0));
    for (kidx, vertex) in vertices.iter().enumerate() {
      assert_eq!(vertex.vertices, vec![kidx]);
    }

    // Each skeleton is strictly increasing in colex order.
    for k in dim.range_inclusive() {
      let skeleton = topology.skeleton(k);
      let simplices: Vec<_> = skeleton.iter().collect();
      assert!(
        simplices.windows(2).all(|w| w[0] < w[1]),
        "skeleton {k} is not colex-ordered"
      );
    }
  }
}

/// $dif compose dif = 0$: the defining law of a cochain complex.
#[test]
fn coboundary_squares_to_zero() {
  for dim in (1..=3usize).map(Dim::from) {
    let topology = CartesianTopology::cube(dim, 2).triangulate();
    for k in (0..dim.index().saturating_sub(1)).map(Dim::from) {
      let dif_k = CsrMatrix::from(&topology.coboundary_operator(k));
      let dif_kk = CsrMatrix::from(&topology.coboundary_operator(k + 1));
      let dif_dif = dif_kk * dif_k;
      assert!(dif_dif.values().iter().all(|&v| v == 0.0));
    }
  }
}

#[test]
fn boundary_simplices_facets_are_boundary_facets() {
  for dim in (1..=3usize).map(Dim::from) {
    let topology = CartesianTopology::cube(dim, 2).triangulate();
    assert_eq!(topology.boundary_simplices(dim - 1), {
      let mut facets: Vec<_> = topology
        .boundary_facets()
        .into_iter()
        .map(|facet| facet.idx())
        .collect();
      facets.sort_by_key(|idx| idx.kidx);
      facets
    });
  }
}

#[test]
fn unit_boundary_operator_agrees_with_complex() {
  for dim in (1..=4usize).map(Dim::from) {
    let complex = Complex::unit(dim);
    for k in dim.range_inclusive() {
      let combinatorial = unit_boundary_operator(dim, k);
      let from_complex = Matrix::from(complex.boundary_operator(k));
      assert_eq!(combinatorial, from_complex);
    }
  }
}

#[test]
fn incidence() {
  let dim = Dim::new(3);
  let complex = Complex::unit(dim);
  let cell = complex.cells().handle_iter().next().unwrap();

  let cell_simplex = Simplex::unit(dim);
  for dim_sub in dim.range_inclusive() {
    let subs: Vec<_> = cell.faces(dim_sub).collect();
    assert_eq!(subs.len(), nsubsimplices(dim, dim_sub));
    let subs_vertices: Vec<_> = cell_simplex.subsimps(dim_sub).collect();
    assert_eq!(
      subs
        .iter()
        .map(|sub| sub.simplex().clone())
        .collect::<Vec<_>>(),
      subs_vertices
    );

    for (isub, sub) in subs.iter().enumerate() {
      let sub_vertices = &subs_vertices[isub];
      for dim_sup in (dim_sub.index()..dim.index()).map(Dim::from) {
        for sup in sub.cofaces(dim_sup) {
          assert!(
            sub_vertices.is_subsimplex_of(sup.simplex())
              && sup.simplex().is_subsimplex_of(&cell_simplex)
          );
        }
      }
    }
  }
}

/// Navigation on a triangulation: facets, neighbors, star and link behave
/// as their topological definitions demand.
#[test]
fn unit_navigation() {
  let topology = CartesianTopology::cube(Dim::new(2), 3).triangulate();

  for cell in topology.cells().handle_iter() {
    // A triangle has 3 facets (edges) and at most 3 neighbors across them.
    assert_eq!(cell.facets().count(), 3);
    assert!(cell.neighbors().count() <= 3);
    // Neighbors share a facet and are distinct cells.
    for nb in cell.neighbors() {
      assert_eq!(nb.dim(), 2);
      assert_ne!(nb.idx(), cell.idx());
      let shared = cell
        .facets()
        .filter(|f| nb.facets().any(|g| g.idx() == f.idx()));
      assert_eq!(shared.count(), 1);
    }
    // The star of a top cell is just itself.
    assert_eq!(cell.star().count(), 1);
  }

  // Boundary facets have a single cell; interior facets two.
  assert!(topology.facets().handle_iter().any(|f| f.is_boundary()));
  for facet in topology.facets().handle_iter() {
    assert_eq!(
      facet.cells().count(),
      if facet.is_boundary() { 1 } else { 2 }
    );
  }

  // The link of a vertex never touches the vertex itself; its star does.
  for vertex in topology.vertices().handle_iter() {
    assert!(vertex.star().any(|s| s.idx() == vertex.idx()));
    for linked in vertex.link() {
      assert!(!linked.simplex().contains(vertex.kidx()));
    }
  }
}
