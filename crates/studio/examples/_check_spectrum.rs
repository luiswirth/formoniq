use formoniq_studio::scene::Scene;
use std::time::Instant;
fn main() {
  for nsub in [3usize, 4] {
    let t = Instant::now();
    let scene = Scene::spherical_harmonics(nsub, 10);
    let dt = t.elapsed();
    let nverts = scene.coords.nvertices();
    println!("subdiv {nsub}: n={nverts} verts, solve {:.2?}", dt);
    for f in &scene.fields {
      let (lo, hi) = f.bounds();
      println!("    {:28}  range=[{lo:+.3},{hi:+.3}]", f.name);
    }
  }
}
