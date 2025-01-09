extern crate nalgebra as na;

use formoniq::fe::whitney::ref_whitney;
use topology::simplex::Simplex;

use std::io::Write;

fn main() {
  let subs = 30;

  let file = std::fs::File::create("out/whitney.txt").unwrap();
  let mut writer = std::io::BufWriter::new(file);

  for x in 0..subs {
    for y in 0..(subs - 1 - x) {
      let xf = x as f64 / (subs - 1) as f64;
      let yf = y as f64 / (subs - 1) as f64;
      let coord = na::DVector::from_row_slice(&[xf, yf]);
      let simplex = Simplex::new(vec![0, 2]);
      let value = ref_whitney(coord.as_view(), simplex).into_1vector();
      let (vx, vy) = (value[0], value[1]);
      writeln!(writer, "{xf:.4} {yf:.4} {vx:.4} {vy:.4}").unwrap();
    }
  }
}
