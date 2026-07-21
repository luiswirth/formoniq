use crate::linalg::Matrix;

use crate::{
  geometry::coord::mesh::{MeshCoords, close_vertex_gaps},
  topology::{
    VertexIdx, complex::Complex, ordering::CellOrdering, simplex::Simplex, skeleton::Skeleton,
  },
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
/// the kind -- element node order is a storage convention, not a structure on
/// the mesh -- so the check is real and a file may well fail it. Refinement
/// falls back to the colex ordering without one
/// ([`Complex::refine`](crate::topology::complex::Complex::refine)), which is
/// always available and always valid; what is lost is only that a refinement
/// tower composes.
///
/// The ordering's parity is separately the winding, reachable through
/// [`CellOrdering::induced_orientation`].
pub fn gmsh2coord_complex_ordered(bytes: &[u8]) -> (Complex, MeshCoords, Option<CellOrdering>) {
  let (cells, coords, words) = gmsh2coord_cells_ordered(bytes);
  let complex = Complex::from_cells(cells);
  // A file may list a cell twice, which the skeleton dedups; then the words do
  // not name the cells one for one and there is no ordering to speak of.
  let ordering = (words.len() == complex.cells().len()).then(|| {
    let ordering = CellOrdering::new(&complex, words);
    ordering.is_face_consistent(&complex).then_some(ordering)
  });
  (complex, coords, ordering.flatten())
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

  let mut points = Vec::new();
  let mut edges = Vec::new();
  let mut trias = Vec::new();
  let mut quads = Vec::new();

  let elements = msh.data.elements.unwrap();
  for block in elements.element_blocks {
    type ElType = mshio::ElementType;
    let simplex_acc = match block.element_type {
      ElType::Pnt => &mut points,
      ElType::Lin2 => &mut edges,
      ElType::Tri3 => &mut trias,
      ElType::Tet4 => &mut quads,
      _ => {
        tracing::warn!("unsupported gmsh ElementType: {:?}", block.element_type);
        continue;
      }
    };
    for e in block.elements {
      // The file's node order, kept: it is the ordering datum, and its parity
      // is the winding. The stored simplex sorts it.
      let word: Vec<VertexIdx> = e.nodes.iter().map(|tag| *tag as usize - 1).collect();
      simplex_acc.push(word);
    }
  }

  let cells = if !quads.is_empty() {
    quads
  } else if !trias.is_empty() {
    trias
  } else if !edges.is_empty() {
    edges
  } else {
    panic!("Failed to construct Triangulation from gmsh.");
  };

  // Gmsh may carry nodes not referenced by any cell; drop them and renumber.
  // The words are relabelled by the same map, so the ordering survives the
  // renumbering rather than being invalidated by it.
  let simplices: Vec<Simplex> = cells
    .iter()
    .map(|word| Simplex::from_word(word.clone()).1)
    .collect();
  let mut used: Vec<VertexIdx> = simplices.iter().flat_map(|cell| cell.iter()).collect();
  used.sort_unstable();
  used.dedup();
  let words = cells
    .into_iter()
    .map(|word| {
      word
        .into_iter()
        .map(|v| used.binary_search(&v).expect("vertex is used"))
        .collect()
    })
    .collect();

  let (cells, mesh_vertices) = close_vertex_gaps(simplices, &mesh_vertices);
  (Skeleton::new(cells), mesh_vertices, words)
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
  /// winds `Pos` -- and the two together are coherent, which is exactly what
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
