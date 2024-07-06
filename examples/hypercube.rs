use formoniq::mesh::factory::hypercube_mesh;

fn main() {
  let d = 3;
  for k in 0..5 {
    let expk = 2usize.pow(k);
    let mesh = hypercube_mesh(d, expk, 1.0);
    println!("{}", mesh.mesh_width());
  }
}
