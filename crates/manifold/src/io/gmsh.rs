use common::linalg::nalgebra::Matrix;

use crate::{
  geometry::coord::MeshVertexCoords,
  topology::{complex::Complex, simplex::Simplex, skeleton::Skeleton},
};

pub fn gmsh2coord_complex(bytes: &[u8]) -> (Complex, MeshVertexCoords) {
  let (cells, coords) = gmsh2coord_cells(bytes);
  let complex = Complex::from_cells(cells);
  (complex, coords)
}

/// Load Gmesh `.msh` file (version 4.1).
pub fn gmsh2coord_cells(bytes: &[u8]) -> (Skeleton, MeshVertexCoords) {
  let msh = mshio::parse_msh_bytes(bytes).unwrap();

  let mesh_vertices = msh.data.nodes.unwrap().node_blocks;
  let mut mesh_vertices: Vec<_> = mesh_vertices
    .iter()
    .flat_map(|block| block.nodes.iter())
    .map(|node| na::dvector![node.x, node.y, node.z])
    .collect();

  if mesh_vertices.iter().all(|coord| coord[2] == 0.0) {
    mesh_vertices
      .iter_mut()
      .for_each(|coord| *coord = na::dvector![coord[0], coord[1]])
  }

  let mesh_vertices = Matrix::from_columns(&mesh_vertices);
  let mesh_vertices = MeshVertexCoords::new(mesh_vertices);

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
      let simplex: Vec<_> = e.nodes.iter().map(|tag| *tag as usize - 1).collect();
      let simplex = Simplex::from(simplex).sorted();
      simplex_acc.push(simplex);
    }
  }

  let skeleton = if !quads.is_empty() {
    quads
  } else if !trias.is_empty() {
    trias
  } else if !edges.is_empty() {
    edges
  } else {
    panic!("Failed to construct Triangulation from gmsh.");
  };

  (Skeleton::new(skeleton), mesh_vertices)
}
