use common::linalg::nalgebra::{Matrix, Vector, VectorView};

use crate::{
  geometry::coord::{mesh::MeshCoords, simplex::SimplexCoords},
  topology::{complex::Complex, simplex::Simplex, skeleton::Skeleton},
};

use std::{collections::HashMap, sync::LazyLock};

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

    // TODO: is this good?
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
/// (e.g. via [`crate::topology::complex::Complex::cells`]) has an
/// essentially arbitrary, alternating winding per triangle. Per-vertex
/// normals of such a list are meaningless without first running this pass:
/// consistent orientation is topological data, recoverable from the manifold's
/// face adjacency, not from geometry (a 2-simplex in $RR^3$ has no signed
/// determinant to read a winding off of).
///
/// Fixes the orientation of one triangle per connected component arbitrarily;
/// the result is correct up to that per-component choice of global sign, which
/// is exactly the ambiguity a normal field has on an orientable surface.
/// Assumes a manifold surface (each edge shared by at most two triangles).
pub fn orient_triangles(triangles: &[[usize; 3]]) -> Vec<[usize; 3]> {
  let edge_key = |a: usize, b: usize| if a < b { (a, b) } else { (b, a) };
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

/// Returns $\[r, theta, phi\]$ with $r in \[0,oo), theta in \[0,pi\], phi in \[0, tau)$
pub fn cartesian2spherical(p: na::Vector3<f64>) -> [f64; 3] {
  let r = p.norm();
  let theta = (p.z / r).acos(); // [0,pi]
  let phi = p.y.atan2(p.x); // [0,tau]
  [r, theta, phi]
}

/// Takes $(r, theta, phi)$ with $r in \[0,oo), theta in \[0,pi\], phi in \[0, tau)$
pub fn spherical2cartesian(r: f64, theta: f64, phi: f64) -> na::Vector3<f64> {
  let x = r * theta.sin() * phi.cos();
  let y = r * theta.sin() * phi.sin();
  let z = r * theta.cos();
  na::Vector3::new(x, y, z)
}

/// Geodesic sphere from subdividing a icosahedron
pub fn mesh_sphere_surface(nsubdivisions: usize) -> TriangleSurface3D {
  let triangles = ICOSAHEDRON_SURFACE.triangles().to_vec();
  let vertex_coords = ICOSAHEDRON_SURFACE
    .vertex_coords()
    .coord_iter()
    .map(|v| v.view().into_owned())
    .collect();

  let (triangles, vertex_coords) = subdivide(triangles, vertex_coords, nsubdivisions);

  TriangleSurface3D::new(triangles, Matrix::from_columns(&vertex_coords))
}

fn subdivide(
  triangles: Vec<[usize; 3]>,
  mut vertex_coords: Vec<Vector>,
  depth: usize,
) -> (Vec<[usize; 3]>, Vec<Vector>) {
  if depth == 0 {
    return (triangles, vertex_coords);
  }

  let mut midpoints = HashMap::new();

  let triangles = triangles
    .into_iter()
    .flat_map(|[v0, v1, v2]| {
      let v01 = get_midpoint(v0, v1, &mut vertex_coords, &mut midpoints);
      let v12 = get_midpoint(v1, v2, &mut vertex_coords, &mut midpoints);
      let v20 = get_midpoint(v2, v0, &mut vertex_coords, &mut midpoints);

      [
        [v0, v01, v20],
        [v1, v12, v01],
        [v2, v20, v12],
        [v01, v12, v20],
      ]
    })
    .collect();

  subdivide(triangles, vertex_coords, depth - 1)
}

fn get_midpoint(
  v0: usize,
  v1: usize,
  vertices: &mut Vec<Vector>,
  midpoints: &mut HashMap<(usize, usize), usize>,
) -> usize {
  let edge = if v0 < v1 { (v0, v1) } else { (v1, v0) };
  if let Some(&midpoint) = midpoints.get(&edge) {
    return midpoint;
  }

  let midpoint = ((&vertices[v0] + &vertices[v1]) / 2.0).normalize();
  vertices.push(midpoint);
  let index = vertices.len() - 1;
  midpoints.insert(edge, index);
  index
}

static ICOSAHEDRON_SURFACE: LazyLock<TriangleSurface3D> = LazyLock::new(|| {
  let phi = f64::midpoint(1.0, 5.0f64.sqrt());

  #[rustfmt::skip]
  let vertex_coords = [
    [-1.0, phi, 0.0],
    [ 1.0, phi, 0.0],
    [-1.0,-phi, 0.0],
    [ 1.0,-phi, 0.0],
    [ 0.0,-1.0, phi],
    [ 0.0, 1.0, phi],
    [ 0.0,-1.0,-phi],
    [ 0.0, 1.0,-phi],
    [ phi, 0.0,-1.0],
    [ phi, 0.0, 1.0],
    [-phi, 0.0,-1.0],
    [-phi, 0.0, 1.0],
  ];

  let vertex_coords: Vec<_> = vertex_coords
    .into_iter()
    .map(|v| na::dvector![v[0], v[1], v[2]].normalize())
    .collect();

  #[rustfmt::skip]
  let triangles = vec![
    [ 0,11, 5],
    [ 0, 5, 1],
    [ 0, 1, 7],
    [ 0, 7,10],
    [ 0,10,11],
    [ 1, 5, 9],
    [ 5,11, 4],
    [11,10, 2],
    [10, 7, 6],
    [ 7, 1, 8],
    [ 3, 9, 4],
    [ 3, 4, 2],
    [ 3, 2, 6],
    [ 3, 6, 8],
    [ 3, 8, 9],
    [ 4, 9, 5],
    [ 2, 4,11],
    [ 6, 2,10],
    [ 8, 6, 7],
    [ 9, 8, 1],
  ];

  TriangleSurface3D::new(triangles, Matrix::from_columns(&vertex_coords))
});
