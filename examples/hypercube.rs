use formoniq::mesh::factory::unit_hypercube_mesh;

fn main() {
  let d = 1;
  let nsubdivisions = 10;
  let mesh = unit_hypercube_mesh(d, nsubdivisions);
  println!("{}", mesh.node_coords());
  for simplex in mesh.dsimplicies(d) {
    println!("{:?}", simplex.vertices());
  }
}
