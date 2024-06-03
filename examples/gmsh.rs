use formoniq::mesh;

use std::fs;

fn main() {
  let msh_bytes = fs::read("res/square.msh").unwrap();
  let mesh = mesh::util::load_gmsh(&msh_bytes);
  println!("{mesh:?}");
}
