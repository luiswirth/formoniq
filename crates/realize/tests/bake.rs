//! Laws for [`realize::bake::BakedMesh`]: the unit cell of every dimension
//! bakes to the primitive $min(n, 2)$ names, the 1-skeleton overlay draws
//! only where there is no fill already, the baked triangles are consistently
//! wound, and a solid bakes only its boundary surface, with every boundary
//! vertex carrying an outward-pointing displacement normal.

use nalgebra as na;
use realize::bake::{BakedMesh, PrimBatch, directed_edges, edge_key};
use regge::coord::mesh::unit_coord_complex;
use std::collections::HashMap;

/// The unit cell of every dimension the ambient reaches bakes, and bakes
/// to the primitive $min(n, 2)$ names: a segment, a triangle, and a
/// tetrahedron's four boundary triangles.
#[test]
fn unit_cell_bakes_at_every_dimension() {
  for dim in 0..=3 {
    let (topology, coords) = unit_coord_complex(dim);
    let coords = coords.embed_euclidean(3);
    let nvertices = topology.nsimplices(0);
    let baked = BakedMesh::new(&topology, &coords);

    assert_eq!(baked.positions.len(), nvertices);
    let ncells = topology.nsimplices(dim);
    match (dim, &baked.cells) {
      (0, PrimBatch::Points(p)) => assert_eq!(p.len(), ncells),
      (1, PrimBatch::Segments(s)) => assert_eq!(s.len(), ncells),
      (2, PrimBatch::Triangles(t)) => assert_eq!(t.len(), ncells),
      // A solid bakes its boundary surface. A single tetrahedron's boundary
      // is its four faces (no face is shared, so all are boundary).
      (3, PrimBatch::Triangles(t)) => assert_eq!(t.len(), 4),
      (d, b) => panic!("dim {d} baked to {b:?}"),
    }
    for &[a, b] in &baked.edges {
      assert!((a as usize) < nvertices && (b as usize) < nvertices);
    }
  }
}

/// The 1-skeleton overlay is the mesh's edges over a filled surface, and empty
/// where the cells already are the 1-skeleton: a curve's edges are drawn
/// once, not twice.
#[test]
fn edges_are_the_overlay_only_where_there_is_a_fill() {
  for dim in 0..=2 {
    let (topology, coords) = unit_coord_complex(dim);
    let coords = coords.embed_euclidean(3);
    let baked = BakedMesh::new(&topology, &coords);
    let expected = if dim == 2 { topology.nsimplices(1) } else { 0 };
    assert_eq!(baked.edges.len(), expected);
  }
}

/// Winding consistency, the property the normals depend on: every edge shared
/// by two triangles is traversed in opposite directions by them. Checked on a
/// closed surface (the tetrahedron's boundary) and on a mesh with boundary
/// (the triforce), where an edge with one incident triangle constrains
/// nothing.
#[test]
fn baked_triangles_are_consistently_wound() {
  let (tet, tet_coords) = unit_coord_complex(3);
  let cases = [
    (tet.clone(), tet_coords.embed_euclidean(3)),
    regge::mesher::teaching::triforce(),
  ];
  for (topology, coords) in cases {
    let baked = BakedMesh::new(&topology, &coords);
    let PrimBatch::Triangles(triangles) = &baked.cells else {
      panic!("a 2- or 3-manifold bakes to triangles");
    };
    let mut seen: HashMap<(u32, u32), (u32, u32)> = HashMap::new();
    for &t in triangles {
      for (a, b) in directed_edges(t) {
        if let Some(&(pa, pb)) = seen.get(&edge_key(a, b)) {
          assert_eq!(
            (pa, pb),
            (b, a),
            "edge ({a}, {b}) traversed the same way by both its triangles"
          );
        } else {
          seen.insert(edge_key(a, b), (a, b));
        }
      }
    }
  }
}

/// A solid bakes its boundary surface, not its full 2-skeleton: the fill is
/// the facets bounding one cell, so a multi-cell grid draws strictly fewer
/// triangles than its interior-inclusive 2-skeleton. Every boundary vertex
/// gets a nonzero displacement normal, and, the boundary being closed,
/// one that points outward, so a constant mode inflates rather than
/// collapses.
#[test]
fn a_solid_bakes_only_its_boundary() {
  use regge::mesher::cartesian::CartesianGrid;
  let (topology, coords) = CartesianGrid::new_unit(3, 3).triangulate();
  let coords = coords.embed_euclidean(3);
  let baked = BakedMesh::new(&topology, &coords);

  let PrimBatch::Triangles(triangles) = &baked.cells else {
    panic!("a solid bakes to triangles");
  };
  let nboundary = topology.boundary_facets().len();
  assert_eq!(triangles.len(), nboundary);
  assert!(
    nboundary < topology.nsimplices(2),
    "the interior was dropped"
  );

  // The unit cube's centroid. A boundary vertex's outward normal has a
  // positive component along the ray from it.
  let centroid = na::Vector3::repeat(0.5);
  let boundary = topology.boundary_complex().unwrap();
  for &v in boundary.parent_kidxs(simplicial::Dim::ZERO) {
    let p = &baked.positions[v];
    let n = na::Vector3::new(p.normal[0] as f64, p.normal[1] as f64, p.normal[2] as f64);
    assert!(
      n.norm() > 0.5,
      "boundary vertex {v} has no displacement normal"
    );
    let pos = na::Vector3::new(
      p.position[0] as f64,
      p.position[1] as f64,
      p.position[2] as f64,
    );
    assert!(
      (pos - centroid).dot(&n) > 0.0,
      "boundary vertex {v} normal points inward"
    );
  }
}
