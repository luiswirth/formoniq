extern crate nalgebra as na;

use formoniq::geometry::CoordSimplex;

fn main() {
  tracing_subscriber::fmt::init();

  for d in 1..=3 {
    let elmat = CoordSimplex::new_ref(d).elmat();
    println!("RefCell in {d} dim has ElMat:");
    println!("{elmat}");
  }
}
