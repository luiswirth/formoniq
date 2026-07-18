//! A tolerant Wavefront OBJ reader for surface meshes.
//!
//! Enough of the format to load a triangulated surface out of the wild -- the
//! built-in gallery's own assets and a mesh the user picks off disk alike. It
//! reads vertex positions and faces and ignores everything else: texture
//! coordinates, normals, materials, groups, smoothing and comments are skipped;
//! a face vertex may carry `v`, `v/vt`, `v/vt/vn` or `v//vn` references and only
//! the position index is taken; an index may be negative (relative to the
//! current end of the vertex list), per the spec; and a polygon of more than
//! three vertices is fan-triangulated.
//!
//! Fallible where a naive reader would panic: a malformed line, an out-of-range
//! index, an empty or non-manifold surface is an [`ObjError`], not a crash, so a
//! file the user picked that is not what it claims is reported rather than
//! taking the viewer down with it.

use std::collections::HashMap;
use std::fmt;

use formoniq_linalg::nalgebra::Matrix;
use simplicial::{geometry::coord::mesh::MeshCoords, topology::complex::Complex};

use crate::io::surface::TriangleSurface3D;

/// Why an OBJ string could not be read as a surface mesh.
#[derive(Debug)]
pub enum ObjError {
  /// A `v`/`f` line whose fields did not parse as the format requires.
  Malformed { line: usize, reason: String },
  /// A face referenced a vertex index outside the vertices declared so far.
  IndexOutOfRange { line: usize, index: isize },
  /// No faces were found: a point cloud, the wrong kind of file, or an
  /// unresolved git-LFS pointer standing in for a not-yet-fetched asset.
  Empty,
  /// An edge shared by three or more triangles: the surface is not a
  /// 2-manifold, which the eigensolve and the renderer both assume.
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
        "no faces found (a point cloud, the wrong file, or an unfetched git-LFS pointer)"
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

  if triangles.is_empty() {
    return Err(ObjError::Empty);
  }
  check_manifold(&triangles)?;

  let columns: Vec<_> = positions
    .iter()
    .map(|p| na::dvector![p[0], p[1], p[2]])
    .collect();
  let coords = MeshCoords::from(Matrix::from_columns(&columns));
  Ok(TriangleSurface3D::new(triangles, coords).into_coord_complex())
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
    // OBJ indices are 1-based; a negative index counts back from the current
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
/// never three or more. The eigensolve assembles against a manifold and the
/// renderer's normals assume one, so a hinge of three-plus triangles is caught
/// here rather than producing a silently wrong spectrum.
fn check_manifold(triangles: &[[usize; 3]]) -> Result<(), ObjError> {
  let mut incidence: HashMap<(usize, usize), usize> = HashMap::new();
  for t in triangles {
    for &(a, b) in &[(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
      let edge = if a <= b { (a, b) } else { (b, a) };
      *incidence.entry(edge).or_insert(0) += 1;
    }
  }
  match incidence.iter().find(|(_, &count)| count > 2) {
    Some((&edge, &count)) => Err(ObjError::NonManifold { edge, count }),
    None => Ok(()),
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// A single triangle with texture/normal references and a trailing comment
  /// reads as one face on three vertices -- the `v/vt/vn` groups and the `#`
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

  /// A file with no faces (a point cloud, or an unfetched LFS pointer) is
  /// reported empty rather than yielding a degenerate mesh.
  #[test]
  fn rejects_faceless_input() {
    assert!(matches!(parse("v 0 0 0\nv 1 0 0\n"), Err(ObjError::Empty)));
  }
}
