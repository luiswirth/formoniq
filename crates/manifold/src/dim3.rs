use crate::{
  geometry::coord::{local::SimplexCoords, VertexCoords},
  topology::{complex::Complex, simplex::Simplex, skeleton::Skeleton},
};

use std::{collections::HashMap, sync::LazyLock};

pub type TriangleTopology = Vec<[usize; 3]>;

#[derive(Debug, Clone)]
pub struct TriangleSurface3D {
  triangles: TriangleTopology,
  coords: VertexCoords,
}
impl TriangleSurface3D {
  pub fn new(triangles: TriangleTopology, coords: impl Into<VertexCoords>) -> Self {
    let coords = coords.into();
    Self { triangles, coords }
  }
  pub fn triangles(&self) -> &[[usize; 3]] {
    &self.triangles
  }
  pub fn vertex_coords(&self) -> &VertexCoords {
    &self.coords
  }
  pub fn vertex_coords_mut(&mut self) -> &mut VertexCoords {
    &mut self.coords
  }
  pub fn into_parts(self) -> (TriangleTopology, VertexCoords) {
    (self.triangles, self.coords)
  }
}

impl TriangleSurface3D {
  pub fn from_coord_skeleton(skeleton: Skeleton, coords: VertexCoords) -> Self {
    assert!(skeleton.dim() == 2, "Topology is not 2D.");
    assert!(coords.dim() <= 3, "Skeleton is not embeddable in 3D.");
    let coords = coords.embed_euclidean(3);

    // TODO: is this good?
    let triangles = skeleton
      .into_index_set()
      .into_iter()
      .map(|simp| {
        let mut vertices: [usize; 3] = simp.vertices.clone().try_into().unwrap();
        let coord_simp = SimplexCoords::from_mesh_simplex(&simp, &coords);
        if coord_simp.orientation().is_neg() {
          vertices.swap(1, 2);
        }
        vertices
      })
      .collect();

    Self::new(triangles, coords)
  }

  pub fn into_coord_skeleton(self) -> (Skeleton, VertexCoords) {
    let simps = self
      .triangles
      .into_iter()
      .map(|tria| Simplex::from(tria).sorted())
      .collect();
    let skeleton = Skeleton::new(simps);
    let coords = self.coords;
    (skeleton, coords)
  }

  pub fn into_coord_complex(self) -> (Complex, VertexCoords) {
    let (skeleton, coords) = self.into_coord_skeleton();
    let complex = Complex::from_cells(skeleton);
    (complex, coords)
  }

  pub fn displace_normal<'a>(&mut self, displacements: impl Into<na::DVectorView<'a, f64>>) {
    let displacements = displacements.into();

    let mut vertex_normals = vec![na::Vector3::zeros(); self.coords.nvertices()];
    let mut vertex_triangle_counts = vec![0; self.coords.nvertices()];
    for ivs in &self.triangles {
      let vs = ivs.map(|i| self.coords.coord(i));
      let e0 = vs[1] - vs[0];
      let e1 = vs[2] - vs[0];
      let triangle_normal = e0.cross(&e1).normalize();
      for &iv in ivs {
        vertex_normals[iv] += &triangle_normal;
        vertex_triangle_counts[iv] += 1;
      }
    }
    for (vertex_normal, count) in vertex_normals.iter_mut().zip(vertex_triangle_counts) {
      *vertex_normal /= count as f64;
    }
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

/// Returns $[r, theta, phi]$ with $r in [0,oo), theta in [0,pi], phi in [0, tau)$
pub fn cartesian2spherical(p: na::Vector3<f64>) -> [f64; 3] {
  let r = p.norm();
  let theta = (p.z / r).acos(); // [0,pi]
  let phi = p.y.atan2(p.x); // [0,tau]
  [r, theta, phi]
}

/// Takes $(r, theta, phi)$ with $r in [0,oo), theta in [0,pi], phi in [0, tau)$
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
    .map(|c| c.into_owned())
    .collect();

  let (triangles, vertex_coords) = subdivide(triangles, vertex_coords, nsubdivisions);

  TriangleSurface3D::new(triangles, vertex_coords.as_slice())
}

fn subdivide(
  triangles: Vec<[usize; 3]>,
  mut vertex_coords: Vec<na::DVector<f64>>,
  depth: usize,
) -> (Vec<[usize; 3]>, Vec<na::DVector<f64>>) {
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
  vertices: &mut Vec<na::DVector<f64>>,
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
  let phi = (1.0 + 5.0f64.sqrt()) / 2.0;

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

  TriangleSurface3D::new(triangles, vertex_coords.as_slice())
});
