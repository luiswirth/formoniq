use formoniq::mesh::factory::unit_hypercube_mesh;

fn main() {
  let d = 3;
  let nsubdivisions = 1;
  let mesh = unit_hypercube_mesh(d, nsubdivisions);
  for simplex in mesh.dsimplicies(d) {
    println!("{:?}", simplex.vertices());
  }
}
