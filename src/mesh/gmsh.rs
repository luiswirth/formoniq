use crate::{
  combinatorics::{OrderedVertplex, Orientation, OrientedVertplex},
  mesh::coordinates::{CoordManifold, NodeCoords},
};
use tracing::warn;

/// Load Gmesh `.msh` file (version 4.1).
pub fn gmsh2coord_mesh(bytes: &[u8]) -> CoordManifold {
  let msh = mshio::parse_msh_bytes(bytes).unwrap();

  let mesh_nodes = msh.data.nodes.unwrap().node_blocks;
  let mesh_nodes: Vec<_> = mesh_nodes
    .iter()
    .flat_map(|block| block.nodes.iter())
    .map(|node| na::DVector::from_column_slice(&[node.x, node.y, node.z]))
    .collect();
  let mesh_nodes = na::DMatrix::from_columns(&mesh_nodes);
  let mesh_nodes = NodeCoords::new(mesh_nodes);

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
      let simplex = e.nodes.iter().map(|tag| *tag as usize - 1).collect();
      let simplex = OrderedVertplex::new(simplex);
      // NOTE: gmsh always produces positively oriented cells
      // TODO: only assume Pos for cells(!) not all simplicies.
      let simplex = OrientedVertplex::new(simplex, Orientation::Pos);
      simplex_acc.push(simplex);
    }
  }

  if !quads.is_empty() {
    return CoordManifold::new(quads, mesh_nodes);
  }
  if !trias.is_empty() {
    return CoordManifold::new(trias, mesh_nodes);
  }
  if !edges.is_empty() {
    return CoordManifold::new(edges, mesh_nodes);
  }
  panic!("failed to construct Triangulation from gmsh");
}
