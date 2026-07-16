//! A triangle surface embedded in $RR^3$: the rendering- and export-facing
//! counterpart of a 2D `manifold::Complex`.
//!
//! This is deliberately dimension-specific and coordinate-full -- exactly the
//! two things formoniq's core keeps out. A `Complex`'s cells carry no winding
//! and no embedding; a graphics API (and a `.obj`/`.mdd` file) needs both,
//! fixed at 3.

use std::collections::HashMap;

use common::linalg::nalgebra::VectorView;
use manifold::{
  geometry::coord::{mesh::MeshCoords, simplex::SimplexCoords},
  topology::{complex::Complex, simplex::Simplex, skeleton::Skeleton},
};

pub type TriangleTopology = Vec<[usize; 3]>;

#[derive(Debug, Clone)]
pub struct TriangleSurface3D {
  triangles: TriangleTopology,
  coords: MeshCoords,
}
impl TriangleSurface3D {
  pub fn new(triangles: TriangleTopology, coords: impl Into<MeshCoords>) -> Self {
    let coords = coords.into();
    Self { triangles, coords }
  }
  pub fn triangles(&self) -> &[[usize; 3]] {
    &self.triangles
  }
  pub fn vertex_coords(&self) -> &MeshCoords {
    &self.coords
  }
  pub fn vertex_coords_mut(&mut self) -> &mut MeshCoords {
    &mut self.coords
  }
  pub fn into_parts(self) -> (TriangleTopology, MeshCoords) {
    (self.triangles, self.coords)
  }
}

impl TriangleSurface3D {
  pub fn from_coord_skeleton(skeleton: Skeleton, coords: MeshCoords) -> Self {
    assert!(skeleton.dim() == 2, "Topology is not 2D.");
    assert!(coords.dim() <= 3, "Skeleton is not embeddable in 3D.");
    let coords = coords.embed_euclidean(3);

    let triangles = skeleton
      .into_index_set()
      .into_iter()
      .map(|simp| {
        let mut vertices: [usize; 3] = simp.clone().try_into().unwrap();
        let coord_simp = SimplexCoords::from_simplex_and_coords(&simp, &coords);
        if coord_simp.orientation().is_neg() {
          vertices.swap(1, 2);
        }
        vertices
      })
      .collect();

    Self::new(triangles, coords)
  }

  pub fn into_coord_skeleton(self) -> (Skeleton, MeshCoords) {
    let simps = self
      .triangles
      .into_iter()
      .map(|tria| Simplex::from_word(tria.to_vec()).1)
      .collect();
    let skeleton = Skeleton::new(simps);
    let coords = self.coords;
    (skeleton, coords)
  }

  pub fn into_coord_complex(self) -> (Complex, MeshCoords) {
    let (skeleton, coords) = self.into_coord_skeleton();
    let complex = Complex::from_cells(skeleton);
    (complex, coords)
  }

  pub fn displace_normal<'a>(&mut self, displacements: impl Into<VectorView<'a>>) {
    let displacements = displacements.into();
    let vertex_normals = vertex_normals(&self.triangles, &self.coords);
    for ((mut v, n), &d) in self
      .coords
      .coord_iter_mut()
      .zip(vertex_normals)
      .zip(displacements)
    {
      v += d * n;
    }
  }
}

/// Consistently winds a triangle soup by flood-filling face adjacency: two
/// triangles sharing an edge are consistent iff they traverse that edge in
/// opposite directions.
///
/// A `Complex`'s cells carry no winding -- vertices are colex-sorted, per the
/// crate's one indexing convention -- so a triangle list read off a `Complex`
/// (e.g. via [`manifold::topology::complex::Complex::cells`]) has an
/// essentially arbitrary, alternating winding per triangle. Per-vertex normals
/// of such a list are meaningless without first running this pass: consistent
/// orientation is topological data, recoverable from the manifold's face
/// adjacency, not from geometry (a 2-simplex in $RR^3$ has no signed
/// determinant to read a winding off of).
///
/// A canonical undirected-edge key: the vertex pair sorted so `(a, b)` and
/// `(b, a)` hash identically. Shared by every edge-adjacency computation in
/// this file.
fn edge_key(a: usize, b: usize) -> (usize, usize) {
  if a < b {
    (a, b)
  } else {
    (b, a)
  }
}

/// Fixes the orientation of one triangle per connected component arbitrarily;
/// the result is correct up to that per-component choice of global sign, which
/// is exactly the ambiguity a normal field has on an orientable surface.
/// Assumes a manifold surface (each edge shared by at most two triangles).
pub fn orient_triangles(triangles: &[[usize; 3]]) -> Vec<[usize; 3]> {
  let directed_edges = |t: [usize; 3]| [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])];

  let mut edge_to_tris: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
  for (ti, &t) in triangles.iter().enumerate() {
    for (a, b) in directed_edges(t) {
      edge_to_tris.entry(edge_key(a, b)).or_default().push(ti);
    }
  }

  let mut oriented = triangles.to_vec();
  let mut visited = vec![false; triangles.len()];
  for start in 0..triangles.len() {
    if visited[start] {
      continue;
    }
    visited[start] = true;
    let mut stack = vec![start];
    while let Some(ti) = stack.pop() {
      for (a, b) in directed_edges(oriented[ti]) {
        for &tj in &edge_to_tris[&edge_key(a, b)] {
          if visited[tj] {
            continue;
          }
          // Consistent winding traverses a shared edge in opposite
          // directions; if `tj` also has the directed edge `(a, b)`, it
          // agrees with `ti` and must be flipped.
          if directed_edges(oriented[tj]).contains(&(a, b)) {
            oriented[tj].swap(1, 2);
          }
          visited[tj] = true;
          stack.push(tj);
        }
      }
    }
  }
  oriented
}

/// Per-vertex normals of a triangle surface embedded in $RR^3$: the average of
/// the unit normals of a vertex's incident triangles.
///
/// Not itself renormalized to unit length -- at a crease or a coarse mesh the
/// average of unit vectors falls short of one -- so a caller after a uniform
/// displacement magnitude should renormalize; [`TriangleSurface3D::displace_normal`]
/// intentionally does not, matching its prior behavior.
pub fn vertex_normals(triangles: &[[usize; 3]], coords: &MeshCoords) -> Vec<na::Vector3<f64>> {
  let mut vertex_normals = vec![na::Vector3::zeros(); coords.nvertices()];
  let mut vertex_triangle_counts = vec![0; coords.nvertices()];
  for ivs in triangles {
    let vs = ivs.map(|i| coords.coord(i));
    let e0 = vs[1] - vs[0];
    let e1 = vs[2] - vs[0];
    let triangle_normal = e0.cross(&e1).normalize();
    for &iv in ivs {
      vertex_normals[iv] += &triangle_normal;
      vertex_triangle_counts[iv] += 1;
    }
  }
  for (vertex_normal, count) in vertex_normals.iter_mut().zip(vertex_triangle_counts) {
    *vertex_normal /= f64::from(count);
  }
  vertex_normals
}
