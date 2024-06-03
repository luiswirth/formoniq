use super::{MeshNodes, Simplex, Skeleton, Triangulation};

/// Load Gmesh `.msh` file (version 4.1).
pub fn load_gmsh(bytes: &[u8]) -> Triangulation {
  let msh = mshio::parse_msh_bytes(bytes).unwrap();

  let nodes = msh.data.nodes.unwrap();
  let nodes = nodes
    .node_blocks
    .iter()
    .flat_map(|block| block.nodes.iter())
    .map(|node| na::DVector::from_column_slice(&[node.x, node.y]))
    .collect();
  let mesh_nodes = MeshNodes::new(nodes);

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
      _ => continue,
    };
    for e in block.elements {
      let indicies = e.nodes.iter().map(|tag| *tag as usize - 1).collect();
      let simplex = Simplex::new(mesh_nodes.clone(), indicies);
      simplex_acc.push(simplex);
    }
  }

  let mut skeletons = Vec::new();
  skeletons.push(Skeleton::new(points));
  if !edges.is_empty() {
    skeletons.push(Skeleton::new(edges));
  }
  if !trias.is_empty() {
    skeletons.push(Skeleton::new(trias));
  }
  if !quads.is_empty() {
    skeletons.push(Skeleton::new(quads));
  }

  Triangulation::from_skeletons(mesh_nodes, skeletons)
}
