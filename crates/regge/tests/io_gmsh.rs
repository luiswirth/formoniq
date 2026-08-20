//! Reading a gmsh `.msh` file: a malformed one is an error, not a panic,
//! and a well-formed one's node order survives as a face-consistent
//! [`CellOrdering`](simplicial::topology::ordering::CellOrdering).

use multiindex::Sign;
use regge::io::gmsh::{gmsh2coord_complex, gmsh2coord_complex_ordered};

/// Two counterclockwise triangles of the unit square, in ASCII `.msh` 4.1.
/// A text literal rather than a fixture file: small enough to read, and the
/// winding is the point of the test.
const SQUARE: &str = "\
$MeshFormat
4.1 0 8
$EndMeshFormat
$Nodes
1 4 1 4
2 1 0 4
1
2
3
4
0 0 0
1 0 0
0 1 0
1 1 0
$EndNodes
$Elements
1 2 1 2
2 1 2 2
1 1 2 3
2 2 4 3
$EndElements
";

/// A file that is not a mesh is reported, not fatal.
///
/// The point is the *return*: these bytes reach the reader from outside the
/// program (a path a reader typed, an asset that is really an unfetched
/// git-LFS pointer), so failing on them has to be something a caller can
/// catch. A panic here would take down whatever offered the file.
#[test]
fn a_file_that_is_not_a_mesh_is_an_error() {
  for bytes in [
    b"not a mesh at all".as_slice(),
    b"".as_slice(),
    // The header alone: well-formed as far as it goes, then nothing.
    b"$MeshFormat\n4.1 0 8\n$EndMeshFormat\n".as_slice(),
    // A truncated node section, which parses into the file and then runs out.
    b"$MeshFormat\n4.1 0 8\n$EndMeshFormat\n$Nodes\n1 4 1 4\n".as_slice(),
  ] {
    assert!(gmsh2coord_complex(bytes).is_err());
  }
}

/// The file's node order survives the read, the renumbering and the colex
/// sort: it is recovered as a face-consistent [`CellOrdering`], and its parity
/// is the winding the file intends.
///
/// The second triangle is stored as ${1, 2, 3}$ but written $(1, 3, 2)$, an
/// odd permutation, so it winds `Neg` against its colex frame while the first
/// winds `Pos`, and the two together are coherent, which is exactly what
/// consistently counterclockwise faces mean.
#[test]
fn the_files_node_order_survives_as_an_ordering() {
  let (complex, coords, ordering) =
    gmsh2coord_complex_ordered(SQUARE.as_bytes()).expect("a well-formed .msh reads");
  assert_eq!(complex.cells().len(), 2);
  assert_eq!(coords.nvertices(), 4);

  let ordering = ordering.expect("consistently wound triangles are face-consistent");
  assert_eq!(ordering.word_by_kidx(0), [0, 1, 2]);
  assert_eq!(ordering.word_by_kidx(1), [1, 3, 2]);

  let orientation = ordering
    .induced_orientation(&complex)
    .expect("a consistently wound surface is coherently oriented");
  assert_eq!(orientation.signs(), [Sign::Pos, Sign::Neg]);
}
