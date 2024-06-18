use formoniq::mesh;

fn main() {
  tracing_subscriber::fmt::init();

  let msh_bytes = std::fs::read("res/square.msh").unwrap();
  let mesh = mesh::util::load_gmsh(&msh_bytes);
  println!("There are {} nodes in this mesh.", mesh.nodes().len());
  for (i, n) in mesh.nodes().iter().enumerate() {
    println!("Node {}", i);
    println!("{}", n.transpose());
  }

  for d in 0..mesh.nskeletons() {
    let skeleton = &mesh.skeletons()[d];
    let nsimps = skeleton.simplicies().len();
    println!("There are {} simplicies in dimension {}.", nsimps, d);
    for (isimp, simp) in skeleton.simplicies().iter().enumerate() {
      let simp_id = (d, isimp);
      println!("Simplex {:?}", simp_id);
      let vertices = simp.vertices();
      println!("\tVertices: {:?}", vertices);
      let simp_coord = mesh.coordinate_simplex(simp_id);
      let det = simp_coord.det();
      println!("\tVolume: {}", det);
      let childs = mesh.subentities(simp_id);
      println!("\tChildren: {childs:?}")
    }
  }
}
