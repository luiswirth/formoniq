//! Laws for [`regge::io::obj`]: a surface written out as a document parses
//! back to the same mesh, the tolerant reader accepts the slash reference
//! forms, fan-triangulates a polygon, resolves negative indices, drops a
//! degenerate face rather than the whole file, discards an unreferenced
//! vertex, rejects a non-manifold or faceless mesh, and recovers a coherent
//! winding as an orientation only when the file is one.

use regge::{
  coord::mesh::MeshCoords,
  io::obj::{ObjError, parse, parse_wound},
};
use simplicial::{Sign, topology::complex::Complex};

/// A sphere written out as a document parses back to the same mesh: same
/// vertices, same cells, same positions. The winding is not recovered as
/// such, a `Complex` stores its cells colex-sorted, but it survives as the
/// orientation the file's faces induce.
#[test]
fn a_surface_round_trips_through_the_document() {
  let (topology, coords) = regge::mesher::sphere::mesh_sphere_surface(1);

  let (read, read_coords, orientation) = parse_wound(&document(&topology, &coords)).unwrap();
  assert_eq!(read.nsimplices(0), topology.nsimplices(0));
  assert_eq!(read.nsimplices(2), topology.nsimplices(2));
  assert!(orientation.is_some(), "a coherently wound surface");
  for (before, after) in coords.coord_iter().zip(read_coords.coord_iter()) {
    assert!((before.view() - after.view()).norm() < 1e-5);
  }
}

/// The surface as an OBJ: its vertices as `v` lines and its cells as `f`
/// lines, each wound by the manifold's own coherent orientation, which is the
/// winding a file is expected to carry.
fn document(topology: &Complex, coords: &MeshCoords) -> String {
  use std::fmt::Write as _;
  let orientation = topology.orientation().expect("a sphere is orientable");
  let mut obj = String::new();
  for coord in coords.coord_iter() {
    let c = coord.view();
    writeln!(obj, "v {} {} {}", c[0], c[1], c[2]).unwrap();
  }
  for cell in topology.cells().handle_iter() {
    let mut corners = cell.simplex().vertices.clone();
    if orientation.sign(cell) == Sign::Neg {
      corners.swap(0, 1);
    }
    // OBJ indexes vertices from one.
    writeln!(
      obj,
      "f {} {} {}",
      corners[0] + 1,
      corners[1] + 1,
      corners[2] + 1
    )
    .unwrap();
  }
  obj
}

/// A single triangle with texture/normal references and a trailing comment
/// reads as one face on three vertices, the `v/vt/vn` groups and the `#`
/// comment are tolerated, not fatal.
#[test]
fn reads_slash_refs_and_comments() {
  let obj = "\
# a triangle
v 0 0 0
v 1 0 0
v 0 1 0
vt 0 0
vn 0 0 1
f 1/1/1 2/1/1 3/1/1
";
  let (complex, coords) = parse(obj).unwrap();
  assert_eq!(coords.nvertices(), 3);
  assert_eq!(complex.nsimplices(2), 1);
}

/// A quad face fan-triangulates into two triangles sharing the diagonal.
#[test]
fn fan_triangulates_a_quad() {
  let obj = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n";
  let (complex, _) = parse(obj).unwrap();
  assert_eq!(complex.nsimplices(2), 2);
}

/// A negative (relative) face index resolves against the vertices seen so
/// far: `-1` is the last vertex.
#[test]
fn resolves_negative_indices() {
  let obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\n";
  let (complex, _) = parse(obj).unwrap();
  assert_eq!(complex.nsimplices(2), 1);
}

/// Three triangles hinged on one edge is not a 2-manifold and is rejected.
#[test]
fn rejects_nonmanifold_hinge() {
  let obj = "\
v 0 0 0
v 1 0 0
v 0 1 0
v 0 0 1
v 0 -1 0
f 1 2 3
f 1 2 4
f 1 2 5
";
  assert!(matches!(parse(obj), Err(ObjError::NonManifold { .. })));
}

/// A file with no faces (a point cloud, or the wrong kind of file) is
/// reported empty rather than yielding a degenerate mesh.
#[test]
fn rejects_faceless_input() {
  assert!(matches!(parse("v 0 0 0\nv 1 0 0\n"), Err(ObjError::Empty)));
}

/// A face naming a corner twice is no simplex. It is dropped, not fatal, and
/// the surface around it still reads.
#[test]
fn drops_degenerate_faces() {
  let obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 2\nf 1 2 3\n";
  let (complex, _) = parse(obj).unwrap();
  assert_eq!(complex.nsimplices(2), 1);

  // A file of nothing but degenerate faces has no surface at all.
  assert!(matches!(
    parse("v 0 0 0\nv 1 0 0\nf 1 2 2\n"),
    Err(ObjError::Empty)
  ));
}

/// A vertex no face references is discarded and the rest relabeled, so the
/// complex's vertices are the contiguous range it requires, whether the
/// orphan sits before the used vertices or after them.
#[test]
fn discards_orphan_vertices() {
  for obj in [
    "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 9 9 9\nf 1 2 3\n",
    "v 9 9 9\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 2 3 4\n",
  ] {
    let (complex, coords) = parse(obj).unwrap();
    assert_eq!(complex.nsimplices(2), 1);
    assert_eq!(complex.nsimplices(0), 3);
    assert_eq!(coords.nvertices(), 3, "coords follow the relabeling");
  }
}

/// Two consistently wound triangles of the unit square carry a coherent
/// orientation; flipping one face's winding destroys it.
///
/// Winding is read as parity and nothing else, and it is validated rather
/// than trusted, a miswound file yields `None`, not a witness that lies.
#[test]
fn winding_becomes_an_orientation_only_when_coherent() {
  let wound = "\
v 0 0 0
v 1 0 0
v 0 1 0
v 1 1 0
f 1 2 3
f 2 4 3
";
  let (_, _, orientation) = parse_wound(wound).unwrap();
  assert!(
    orientation.is_some(),
    "consistent winding is a coherent orientation"
  );

  // The second face reversed: the two now induce the same orientation on the
  // edge they share.
  let flipped = wound.replace("f 2 4 3", "f 2 3 4");
  let (_, _, orientation) = parse_wound(&flipped).unwrap();
  assert!(
    orientation.is_none(),
    "a flipped face is not coherently wound"
  );
}
