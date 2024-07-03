use formoniq::mesh::factory::unit_hypercube_mesh;

fn main() {
  let d = 3;
  for k in 0..5 {
    let expk = 2usize.pow(k);
    let mesh = unit_hypercube_mesh(d, expk);
    println!("{}", mesh.mesh_width());
  }
}
