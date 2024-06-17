use formoniq::mesh;

use std::fs;

fn main() {
  let msh_bytes = fs::read("res/cube.msh").unwrap();
  let mesh = mesh::util::load_gmsh(&msh_bytes);
  for d in 0..mesh.nskeletons() {
    let skeleton = &mesh.skeletons()[d];
    let nsimps = skeleton.simplicies().len();
    println!("There are {} simplicies in dimension {}.", nsimps, d);
    for isimp in 0..skeleton.nsimplicies() {
      let simp = mesh.coordinate_simplex((d, isimp));
      let det = simp.det();
      println!("Simplex {} has volume {}.", isimp, det);
    }
  }
}
