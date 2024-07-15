use tracing::warn;

use crate::mesh::MeshSimplex;

use super::SimplicialMesh;

/// Load Gmesh `.msh` file (version 4.1).
pub fn load_gmsh(bytes: &[u8]) -> SimplicialMesh {
  let msh = mshio::parse_msh_bytes(bytes).unwrap();

  let mesh_nodes: Vec<_> = msh
    .data
    .nodes
    .unwrap()
    .node_blocks
    .iter()
    .flat_map(|block| block.nodes.iter())
    .map(|node| na::DVector::from_column_slice(&[node.x, node.y, node.z]))
    .collect();
  let mesh_nodes = na::DMatrix::from_columns(&mesh_nodes);

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
        warn!("unsupported gmsh ElementType: {:?}", block.element_type);
        continue;
      }
    };
    for e in block.elements {
      let vertices = e.nodes.iter().map(|tag| *tag as usize - 1).collect();
      let simplex = MeshSimplex::new(vertices);
      simplex_acc.push(simplex);
    }
  }

  if !quads.is_empty() {
    return SimplicialMesh::new(mesh_nodes, quads);
  }
  if !trias.is_empty() {
    return SimplicialMesh::new(mesh_nodes, trias);
  }
  if !edges.is_empty() {
    return SimplicialMesh::new(mesh_nodes, edges);
  }
  panic!("failed to construct Triangulation from gmsh");
}
