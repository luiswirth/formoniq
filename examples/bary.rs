extern crate nalgebra as na;

use formoniq::geometry::CoordSimplex;

fn main() {
  tracing_subscriber::fmt::init();

  let vertices = na::DMatrix::from_column_slice(1, 2, &[0.0, 0.5]);
  let simplex = CoordSimplex::new(vertices);
  let bary = simplex.barycentric_functions_grad();
  println!("{bary}");
}
