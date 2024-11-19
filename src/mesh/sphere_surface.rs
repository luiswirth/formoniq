use crate::VertexIdx;

use super::dim3::TriangleSurface3D;

use std::{collections::HashMap, sync::LazyLock};

/// Geodesic sphere from subdividing a icosahedron
pub fn mesh_sphere_surface(nsubdivisions: usize) -> TriangleSurface3D {
  let triangles = ICOSAHEDRON_SURFACE.triangles().to_vec();
  let node_coords = ICOSAHEDRON_SURFACE
    .node_coords()
    .column_iter()
    .map(|c| c.into_owned())
    .collect();

  let (triangles, node_coords) = subdivide(triangles, node_coords, nsubdivisions);

  let node_coords = na::Matrix3xX::from_columns(&node_coords);

  TriangleSurface3D::new(triangles, node_coords)
}

fn subdivide(
  triangles: Vec<[VertexIdx; 3]>,
  mut node_coords: Vec<na::Vector3<f64>>,
  depth: usize,
) -> (Vec<[VertexIdx; 3]>, Vec<na::Vector3<f64>>) {
  if depth == 0 {
    return (triangles, node_coords);
  }

  let mut midpoints = HashMap::new();

  let triangles = triangles
    .into_iter()
    .flat_map(|[v0, v1, v2]| {
      let v01 = get_midpoint(v0, v1, &mut node_coords, &mut midpoints);
      let v12 = get_midpoint(v1, v2, &mut node_coords, &mut midpoints);
      let v20 = get_midpoint(v2, v0, &mut node_coords, &mut midpoints);

      [
        [v0, v01, v20],
        [v1, v12, v01],
        [v2, v20, v12],
        [v01, v12, v20],
      ]
    })
    .collect();

  subdivide(triangles, node_coords, depth - 1)
}

fn get_midpoint(
  v0: usize,
  v1: usize,
  vertices: &mut Vec<na::Vector3<f64>>,
  midpoints: &mut HashMap<(usize, usize), usize>,
) -> usize {
  let edge = if v0 < v1 { (v0, v1) } else { (v1, v0) };
  if let Some(&midpoint) = midpoints.get(&edge) {
    return midpoint;
  }

  let midpoint = ((vertices[v0] + vertices[v1]) / 2.0).normalize();
  vertices.push(midpoint);
  let index = vertices.len() - 1;
  midpoints.insert(edge, index);
  index
}

static ICOSAHEDRON_SURFACE: LazyLock<TriangleSurface3D> = LazyLock::new(|| {
  let phi = (1.0 + 5.0f64.sqrt()) / 2.0;

  #[rustfmt::skip]
  let vertices = [
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

  let vertices: Vec<_> = vertices
    .into_iter()
    .map(|v| na::Vector3::new(v[0], v[1], v[2]).normalize())
    .collect();
  let node_coords = na::Matrix3xX::from_columns(&vertices);

  let triangles = vec![
    [0, 11, 5],
    [0, 5, 1],
    [0, 1, 7],
    [0, 7, 10],
    [0, 10, 11],
    [1, 5, 9],
    [5, 11, 4],
    [11, 10, 2],
    [10, 7, 6],
    [7, 1, 8],
    [3, 9, 4],
    [3, 4, 2],
    [3, 2, 6],
    [3, 6, 8],
    [3, 8, 9],
    [4, 9, 5],
    [2, 4, 11],
    [6, 2, 10],
    [8, 6, 7],
    [9, 8, 1],
  ];

  TriangleSurface3D::new(triangles, node_coords)
});
