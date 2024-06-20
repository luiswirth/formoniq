extern crate nalgebra as na;

use formoniq::geometry::CoordSimplex;

fn mat_round(m: na::DMatrix<f64>) -> na::DMatrix<f64> {
  const ACCURACY: f64 = 100.0;
  m.map(|v| (v * ACCURACY).round() / ACCURACY)
}

fn main() {
  tracing_subscriber::fmt::init();

  for d in 1..=3 {
    let mut elmat = CoordSimplex::new_ref(d).elmat();
    elmat = mat_round(elmat);
    println!("{d}-RefCell in {d} dim has ElMat:");
    println!("{elmat}");

    let intrinsic_dim = d;
    let ambient_dim = intrinsic_dim + 1;
    let mut elmat = CoordSimplex::new_ref_embedded(intrinsic_dim, ambient_dim).elmat();
    elmat = mat_round(elmat);
    println!("{intrinsic_dim}-RefCell in {ambient_dim} dim has ElMat:");
    println!("{elmat}");
  }
}
