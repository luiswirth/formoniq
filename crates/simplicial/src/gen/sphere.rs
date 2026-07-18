//! Geodesic sphere generation: a 2-manifold embedded in $RR^3$, from
//! subdividing an icosahedron.
//!
//! Unlike [`super::cartesian`], this is not dimension-agnostic -- there is no
//! uniform way to triangulate an $n$-sphere for arbitrary $n$, so this
//! generator is fixed at the one case that is both simple and useful: the
//! 2-sphere, embedded in $RR^3$. It exists as real mesh input (a closed,
//! non-contractible manifold with known Betti numbers $(1, 0, 1)$, used
//! throughout the test suite and by `formoniq-studio`'s spherical-harmonics
//! scene) -- not as a rendering utility.

use std::collections::HashMap;

use formoniq_linalg::nalgebra::{Matrix, Vector};

use crate::{
  geometry::coord::mesh::MeshCoords,
  topology::{complex::Complex, simplex::Simplex, skeleton::Skeleton},
};

/// Geodesic sphere from subdividing an icosahedron `nsubdivisions` times.
pub fn mesh_sphere_surface(nsubdivisions: usize) -> (Complex, MeshCoords) {
  let (triangles, vertex_coords) = subdivide(
    ICOSAHEDRON_TRIANGLES.to_vec(),
    icosahedron_vertices(),
    nsubdivisions,
  );

  // The icosahedron (and every subdivision of it) is consistently wound by
  // construction, so no orientation fix-up is needed: `Skeleton` canonicalizes
  // each cell to its colex vertex order regardless, and carries no separate
  // winding datum for a top cell to get wrong.
  let cells = triangles
    .into_iter()
    .map(|tri| Simplex::from_word(tri.to_vec()).1)
    .collect();
  let skeleton = Skeleton::new(cells);
  let complex = Complex::from_cells(skeleton);
  let coords = MeshCoords::from(Matrix::from_columns(&vertex_coords));
  (complex, coords)
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

#[rustfmt::skip]
const ICOSAHEDRON_TRIANGLES: [[usize; 3]; 20] = [
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

fn icosahedron_vertices() -> Vec<Vector> {
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

  vertex_coords
    .into_iter()
    .map(|v| na::dvector![v[0], v[1], v[2]].normalize())
    .collect()
}
