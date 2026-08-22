//! Reading a surface mesh from a Wavefront OBJ.
//!
//! An OBJ is a wound triangle surface in $RR^3$, and a file from the wild is
//! not a mesh until it has been checked, so the reader is deliberately the
//! tolerant one.
//!
//! It accepts enough of the format to load a triangulated surface. It
//! reads vertex positions and faces and ignores everything else: texture
//! coordinates, normals, materials, groups, smoothing and comments are skipped;
//! a face vertex may carry `v`, `v/vt`, `v/vt/vn` or `v//vn` references and only
//! the position index is taken. An index may be negative (relative to the
//! current end of the vertex list), per the spec; and a polygon of more than
//! three vertices is fan-triangulated.
//!
//! Two habits of a mesh from the wild are repaired rather than refused, because
//! neither changes the surface: a degenerate face, whose corners are not three
//! distinct vertices and which therefore is no simplex, is dropped; and vertices
//! no face references are discarded, the rest relabeled onto the contiguous
//! range a `Complex` requires.
//!
//! Fallible where a naive reader would panic: a malformed line, an out-of-range
//! index, an empty or non-manifold surface is an [`ObjError`], not a crash, so
//! a file that is not what it claims is reported to whoever offered it rather
//! than taking the caller down with it.

use std::collections::HashMap;
use std::fmt;

use crate::coord::mesh::MeshCoords;
use simplicial::linalg::Matrix;
use simplicial::topology::{
  complex::Complex, ordering::CellOrdering, orientation::Orientation, relabel::VertexRelabelling,
  simplex::Simplex, skeleton::Skeleton,
};

/// Why an OBJ string could not be read as a surface mesh.
#[derive(Debug)]
pub enum ObjError {
  /// A `v`/`f` line whose fields did not parse as the format requires.
  Malformed { line: usize, reason: String },
  /// A face referenced a vertex index outside the vertices declared so far.
  IndexOutOfRange { line: usize, index: isize },
  /// No faces were found: a point cloud, or the wrong kind of file.
  Empty,
  /// An edge shared by three or more triangles: the surface is not a
  /// 2-manifold.
  NonManifold { edge: (usize, usize), count: usize },
}

impl fmt::Display for ObjError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      ObjError::Malformed { line, reason } => write!(f, "line {line}: {reason}"),
      ObjError::IndexOutOfRange { line, index } => {
        write!(f, "line {line}: face vertex index {index} out of range")
      }
      ObjError::Empty => write!(
        f,
        "no faces found (a point cloud, or the wrong kind of file)"
      ),
      ObjError::NonManifold { edge, count } => write!(
        f,
        "not a 2-manifold: edge ({}, {}) is shared by {count} triangles",
        edge.0, edge.1
      ),
    }
  }
}

impl std::error::Error for ObjError {}

/// Reads an OBJ string as a triangulated surface `Complex` with its ambient
/// (3D) coordinates. See the module docs for the accepted subset of the format.
pub fn parse(obj: &str) -> Result<(Complex, MeshCoords), ObjError> {
  let (complex, coords, _) = parse_wound(obj)?;
  Ok((complex, coords))
}

/// As [`parse`], also recovering the winding the file's faces are written in,
/// as an [`Orientation`].
///
/// A face's corner order in an OBJ is winding: which way the surface normal
/// points. That is orientation data, not the vertex ordering a refinement
/// inherits, the same-shaped datum meaning a different thing, so it is
/// returned as an `Orientation` and only its parity is read.
///
/// `None` when the file is not consistently wound, and hence carries no
/// coherent orientation: a mesh from the wild may well have flipped faces, and
/// on a non-orientable surface no winding could be coherent at all. The
/// orientation is validated rather than asserted, so the witness still proves
/// orientability.
pub fn parse_wound(obj: &str) -> Result<(Complex, MeshCoords, Option<Orientation>), ObjError> {
  let mut positions: Vec<[f64; 3]> = Vec::new();
  let mut triangles: Vec<[usize; 3]> = Vec::new();

  for (i, raw) in obj.lines().enumerate() {
    let line_no = i + 1;
    // A `#` starts a comment to end of line, anywhere.
    let line = raw.split('#').next().unwrap_or("").trim();
    let mut tokens = line.split_whitespace();
    match tokens.next() {
      Some("v") => positions.push(parse_vertex(tokens, line_no)?),
      Some("f") => {
        let corners = parse_face(tokens, positions.len(), line_no)?;
        // Fan-triangulate: a convex polygon $v_0 v_1 ... v_{m-1}$ splits into
        // triangles $(v_0, v_{w-1}, v_w)$. A triangle passes through unchanged.
        for w in 2..corners.len() {
          triangles.push([corners[0], corners[w - 1], corners[w]]);
        }
      }
      // vt, vn, vp, o, g, s, mtllib, usemtl, blank, comment-only: not geometry.
      _ => {}
    }
  }

  // A face may name the same corner twice, directly, or through a fan of an
  // n-gon that does. The triangle carries no geometry, is not a simplex (its
  // vertices are not distinct), and would otherwise spuriously raise an edge's
  // incidence count, so it is dropped before the surface is judged.
  triangles.retain(|[a, b, c]| a != b && b != c && a != c);
  if triangles.is_empty() {
    return Err(ObjError::Empty);
  }
  check_manifold(&triangles)?;

  // An OBJ from the wild routinely carries loose points, or a `v` block shared
  // by an object whose faces were not exported, while a complex is built on the
  // vertices $0..m$ with every one of them used. Closing the gap here rather
  // than after the fact keeps the triangle list, and hence the winding words
  // read off it, in one numbering throughout.
  let relabelling = VertexRelabelling::of_used(triangles.iter().flatten().copied());
  for corner in triangles.iter_mut().flatten() {
    *corner = relabelling.relabel(*corner);
  }
  let positions: Vec<_> = relabelling.used().iter().map(|&v| positions[v]).collect();

  let columns: Vec<_> = positions
    .iter()
    .map(|p| na::dvector![p[0], p[1], p[2]])
    .collect();
  let coords = MeshCoords::from(Matrix::from_columns(&columns));
  let words: Vec<Vec<usize>> = triangles.iter().map(|t| t.to_vec()).collect();
  let complex = Complex::from_cells(Skeleton::new(
    words
      .iter()
      .map(|t| Simplex::from_word(t.clone()).1)
      .collect(),
  ));

  let orientation = CellOrdering::try_new(&complex, words)
    .and_then(|ordering| ordering.induced_orientation(&complex));
  Ok((complex, coords, orientation))
}

/// The first three whitespace-separated floats of a `v` line; any further
/// fields (a `w` coordinate, or per-vertex colors) are ignored.
fn parse_vertex<'a>(
  tokens: impl Iterator<Item = &'a str>,
  line_no: usize,
) -> Result<[f64; 3], ObjError> {
  let mut coord = [0.0; 3];
  let mut n = 0;
  for (slot, tok) in coord.iter_mut().zip(tokens) {
    *slot = tok.parse::<f64>().map_err(|e| ObjError::Malformed {
      line: line_no,
      reason: format!("vertex coordinate `{tok}`: {e}"),
    })?;
    n += 1;
  }
  if n < 3 {
    return Err(ObjError::Malformed {
      line: line_no,
      reason: "a vertex needs three coordinates".to_string(),
    });
  }
  Ok(coord)
}

/// The resolved 0-based position indices of a face's corners, taking only the
/// position index of each `v/vt/vn` group and resolving negative (relative)
/// indices against `nvertices`, the vertex count seen so far.
fn parse_face<'a>(
  tokens: impl Iterator<Item = &'a str>,
  nvertices: usize,
  line_no: usize,
) -> Result<Vec<usize>, ObjError> {
  let mut corners = Vec::new();
  for spec in tokens {
    let field = spec.split('/').next().unwrap_or("");
    let raw: isize = field.parse().map_err(|e| ObjError::Malformed {
      line: line_no,
      reason: format!("face vertex `{spec}`: {e}"),
    })?;
    // OBJ indices are 1-based. A negative index counts back from the current
    // end of the vertex list ($-1$ is the last vertex).
    let resolved = if raw < 0 {
      nvertices as isize + raw
    } else {
      raw - 1
    };
    if resolved < 0 || resolved as usize >= nvertices {
      return Err(ObjError::IndexOutOfRange {
        line: line_no,
        index: raw,
      });
    }
    corners.push(resolved as usize);
  }
  if corners.len() < 3 {
    return Err(ObjError::Malformed {
      line: line_no,
      reason: "a face needs at least three vertices".to_string(),
    });
  }
  Ok(corners)
}

/// Rejects a triangle soup that is not a 2-manifold: every undirected edge of a
/// surface mesh bounds one triangle (on the boundary) or two (in the interior),
/// never three or more. A hinge of three-plus triangles is caught here rather
/// than reaching a caller that assumes a manifold.
fn check_manifold(triangles: &[[usize; 3]]) -> Result<(), ObjError> {
  let mut incidence: HashMap<(usize, usize), usize> = HashMap::new();
  for t in triangles {
    for &(a, b) in &[(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
      let edge = if a <= b { (a, b) } else { (b, a) };
      *incidence.entry(edge).or_insert(0) += 1;
    }
  }
  match incidence.iter().find(|&(_, &count)| count > 2) {
    Some((&edge, &count)) => Err(ObjError::NonManifold { edge, count }),
    None => Ok(()),
  }
}
