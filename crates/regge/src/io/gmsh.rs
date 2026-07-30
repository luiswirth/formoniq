use simplicial::linalg::Matrix;

use crate::coord::mesh::MeshCoords;
use simplicial::topology::{
  VertexIdx, complex::Complex, ordering::CellOrdering, relabel::VertexRelabelling,
  simplex::Simplex, skeleton::Skeleton,
};

pub fn gmsh2coord_complex(bytes: &[u8]) -> (Complex, MeshCoords) {
  let (cells, coords) = gmsh2coord_cells(bytes);
  let complex = Complex::from_cells(cells);
  (complex, coords)
}

/// Load a `.msh` keeping the node order Gmsh wrote each element in, as a
/// [`CellOrdering`].
///
/// `None` when that ordering is not face-consistent. Gmsh promises nothing of
/// the kind, element node order is a storage convention, not a structure on
/// the mesh, so the check is real and a file may well fail it. Refinement
/// falls back to the colex ordering without one
/// ([`Complex::refine`](simplicial::topology::complex::Complex::refine)), which is
/// always available and always valid. What is lost is only that a refinement
/// tower composes.
///
/// The ordering's parity is separately the winding, reachable through
/// [`CellOrdering::induced_orientation`].
pub fn gmsh2coord_complex_ordered(bytes: &[u8]) -> (Complex, MeshCoords, Option<CellOrdering>) {
  let (cells, coords, words) = gmsh2coord_cells_ordered(bytes);
  let complex = Complex::from_cells(cells);
  let ordering =
    CellOrdering::try_new(&complex, words).filter(|ordering| ordering.is_face_consistent(&complex));
  (complex, coords, ordering)
}

/// Load Gmesh `.msh` file (version 4.1).
pub fn gmsh2coord_cells(bytes: &[u8]) -> (Skeleton, MeshCoords) {
  let (skeleton, coords, _) = gmsh2coord_cells_ordered(bytes);
  (skeleton, coords)
}

/// As [`gmsh2coord_cells`], also returning each cell's nodes in the order the
/// file lists them, renumbered alongside the cells.
pub fn gmsh2coord_cells_ordered(bytes: &[u8]) -> (Skeleton, MeshCoords, Vec<Vec<VertexIdx>>) {
  let msh = mshio::parse_msh_bytes(bytes).unwrap();

  let mesh_vertices = msh.data.nodes.unwrap().node_blocks;
  let mut mesh_vertices: Vec<_> = mesh_vertices
    .iter()
    .flat_map(|block| block.nodes.iter())
    .map(|node| na::dvector![node.x, node.y, node.z])
    .collect();

  if mesh_vertices.iter().all(|coord| coord[2] == 0.0) {
    for coord in &mut mesh_vertices {
      *coord = na::dvector![coord[0], coord[1]];
    }
  }

  let mesh_vertices = Matrix::from_columns(&mesh_vertices);
  let mesh_vertices = MeshCoords::new(mesh_vertices);

  // The elements of each simplex dimension the file carries, as the node words
  // it lists them in: that order is the ordering datum, and its parity is the
  // winding. The stored simplex sorts it.
  let mut by_dim: [Vec<Vec<VertexIdx>>; 4] = Default::default();

  let elements = msh.data.elements.unwrap();
  for block in elements.element_blocks {
    type ElType = mshio::ElementType;
    let dim = match block.element_type {
      ElType::Pnt => 0,
      ElType::Lin2 => 1,
      ElType::Tri3 => 2,
      ElType::Tet4 => 3,
      _ => {
        tracing::warn!("unsupported gmsh ElementType: {:?}", block.element_type);
        continue;
      }
    };
    for e in block.elements {
      by_dim[dim].push(e.nodes.iter().map(|tag| *tag as usize - 1).collect());
    }
  }

  // The cells are the elements of the highest dimension present: a file lists
  // the lower-dimensional ones as its physical groups, and they are faces of
  // the cells rather than cells of their own.
  let cells = by_dim
    .into_iter()
    .rev()
    .find(|elements| !elements.is_empty())
    .expect("a gmsh file must carry elements of some supported type");

  // Gmsh may name nodes no cell references; drop them and renumber. The words
  // are relabelled by the same map, so the ordering survives the renumbering
  // rather than being invalidated by it.
  let relabelling = VertexRelabelling::of_used(cells.iter().flatten().copied());
  let words: Vec<Vec<VertexIdx>> = cells
    .into_iter()
    .map(|word| relabelling.relabel_word(word))
    .collect();
  let simplices = words
    .iter()
    .map(|word| Simplex::from_word(word.clone()).1)
    .collect();

  (
    Skeleton::new(simplices),
    mesh_vertices.select(relabelling.used()),
    words,
  )
}

#[cfg(test)]
mod test {
  use super::*;
  use multiindex::Sign;

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
    let (complex, coords, ordering) = gmsh2coord_complex_ordered(SQUARE.as_bytes());
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
}
