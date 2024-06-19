use formoniq::mesh;

fn main() {
  tracing_subscriber::fmt::init();

  let msh_bytes = std::fs::read("res/square.msh").unwrap();
  let mesh = mesh::gmsh::load_gmsh(&msh_bytes);
  println!("There are {} nodes in this mesh.", mesh.nnodes());
  for (i, n) in mesh.node_coords().column_iter().enumerate() {
    println!("Node {}", i);
    println!("{}", n.transpose());
  }

  for (d, dsimps) in mesh.simplicies().into_iter().enumerate() {
    let ndsimps = dsimps.len();
    println!("There are {} simplicies in dimension {}.", ndsimps, d);
    for (isimp, simp) in dsimps.iter().enumerate() {
      let simp_id = (d, isimp);
      println!("Simplex {:?}", simp_id);
      let vertices = simp.vertices();
      println!("\tVertices: {:?}", vertices);
      let simp_coord = mesh.coordinate_simplex(simp_id);
      let det = simp_coord.det();
      println!("\tVolume: {}", det);
      let childs = mesh.simplex_faces(simp_id);
      println!("\tChildren: {childs:?}")
    }
  }
}
