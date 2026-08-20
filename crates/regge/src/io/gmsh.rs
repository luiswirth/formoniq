//! Gmsh `.msh` (version 4.1) as a complex and its coordinates.
//!
//! A file is external input, so every way one can fail to be a mesh is a
//! [`GmshError`] and never a panic: a file that is not what it claims is
//! reported to whoever offered it, rather than taking the caller down with it.
//!
//! The format caps the dimension at three, its highest simplicial element being
//! the tetrahedron. That is the *format's* limit and not this crate's: a mesh
//! above it simply cannot be written as `.msh`.

use std::fmt;

use simplicial::linalg::Matrix;

use crate::coord::mesh::MeshCoords;
use simplicial::topology::{
  VertexIdx, complex::Complex, ordering::CellOrdering, relabel::VertexRelabelling,
  simplex::Simplex, skeleton::Skeleton,
};

/// Why a `.msh` byte string could not be read as a mesh.
#[derive(Debug)]
pub enum GmshError {
  /// The bytes are not a well-formed `.msh` document.
  Malformed(String),
  /// The document declares no node section, so its elements index nothing.
  NoNodes,
  /// The document declares no element section, so it carries no cells.
  NoElements,
  /// Elements are present, but none of a supported simplicial type: the
  /// format's non-simplicial elements (quadrilaterals, hexahedra, the
  /// higher-order variants) have no place in a simplicial complex.
  NoSupportedElements,
}

impl fmt::Display for GmshError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      GmshError::Malformed(reason) => write!(f, "malformed .msh: {reason}"),
      GmshError::NoNodes => write!(f, "no node section"),
      GmshError::NoElements => write!(f, "no element section"),
      GmshError::NoSupportedElements => {
        write!(f, "no point, line, triangle or tetrahedron elements")
      }
    }
  }
}

impl std::error::Error for GmshError {}

/// Reads a `.msh` as a [`Complex`] with its coordinates.
pub fn gmsh2coord_complex(bytes: &[u8]) -> Result<(Complex, MeshCoords), GmshError> {
  let (cells, coords) = gmsh2coord_cells(bytes)?;
  let complex = Complex::from_cells(cells);
  Ok((complex, coords))
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
pub fn gmsh2coord_complex_ordered(
  bytes: &[u8],
) -> Result<(Complex, MeshCoords, Option<CellOrdering>), GmshError> {
  let (cells, coords, words) = gmsh2coord_cells_ordered(bytes)?;
  let complex = Complex::from_cells(cells);
  let ordering =
    CellOrdering::try_new(&complex, words).filter(|ordering| ordering.is_face_consistent(&complex));
  Ok((complex, coords, ordering))
}

/// Reads a `.msh` as the [`Skeleton`] of its cells with their coordinates.
pub fn gmsh2coord_cells(bytes: &[u8]) -> Result<(Skeleton, MeshCoords), GmshError> {
  let (skeleton, coords, _) = gmsh2coord_cells_ordered(bytes)?;
  Ok((skeleton, coords))
}

/// As [`gmsh2coord_cells`], also returning each cell's nodes in the order the
/// file lists them, renumbered alongside the cells.
pub fn gmsh2coord_cells_ordered(
  bytes: &[u8],
) -> Result<(Skeleton, MeshCoords, Vec<Vec<VertexIdx>>), GmshError> {
  let msh = mshio::parse_msh_bytes(bytes).map_err(|e| GmshError::Malformed(format!("{e}")))?;

  let mesh_vertices = msh.data.nodes.ok_or(GmshError::NoNodes)?.node_blocks;
  let mut mesh_vertices: Vec<_> = mesh_vertices
    .iter()
    .flat_map(|block| block.nodes.iter())
    .map(|node| na::dvector![node.x, node.y, node.z])
    .collect();

  // Gmsh writes three coordinates whatever the mesh's dimension, so a planar
  // mesh arrives as one lying in $z = 0$ and the third coordinate is the
  // format's padding rather than an ambient direction. Dropping it is reading
  // that convention, which is why $z$ is asked about and not each axis in turn.
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

  let elements = msh.data.elements.ok_or(GmshError::NoElements)?;
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
    .ok_or(GmshError::NoSupportedElements)?;

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

  Ok((
    Skeleton::new(simplices),
    mesh_vertices.select(relabelling.used()),
    words,
  ))
}
