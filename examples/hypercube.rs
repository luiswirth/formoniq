use formoniq::mesh::factory::unit_hypercube_mesh;

fn main() {
  let d = 2;
  let nsubdivisions = 1;
  let mesh = unit_hypercube_mesh(d, nsubdivisions);
  println!("{mesh:?}");
}
