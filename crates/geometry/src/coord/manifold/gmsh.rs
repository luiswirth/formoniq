use crate::coord::MeshVertexCoords;

use index_algebra::sign::Sign;
use topology::{simplex::Simplex, skeleton::TopologySkeleton};

/// Load Gmesh `.msh` file (version 4.1).
pub fn gmsh2coord_mesh(bytes: &[u8]) -> (TopologySkeleton, MeshVertexCoords) {
  let msh = mshio::parse_msh_bytes(bytes).unwrap();

  let mesh_vertices = msh.data.nodes.unwrap().node_blocks;
  let mesh_vertices: Vec<_> = mesh_vertices
    .iter()
    .flat_map(|block| block.nodes.iter())
    .map(|node| na::DVector::from_column_slice(&[node.x, node.y, node.z]))
    .collect();
  let mesh_vertices = na::DMatrix::from_columns(&mesh_vertices);
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
      // NOTE: gmsh always produces positively oriented cells
      // TODO: only assume Pos for cells(!) not all simplicies.
      let simplex = Simplex::from(simplex).with_sign(Sign::Pos);
      simplex_acc.push(simplex);
    }
  }

  // TODO: create complex from all given simplicies (also subs)
  let skeleton = if !quads.is_empty() {
    quads
  } else if !trias.is_empty() {
    trias
  } else if !edges.is_empty() {
    edges
  } else {
    panic!("Failed to construct Triangulation from gmsh.");
  };

  let skeleton = skeleton
    .into_iter()
    .map(|simp| simp.into_simplex().into_sorted())
    .collect();
  let skeleton = TopologySkeleton::new(skeleton);

  (skeleton, mesh_vertices)
}
