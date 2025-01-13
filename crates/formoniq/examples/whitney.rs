extern crate nalgebra as na;

use formoniq::fe::whitney::ref_whitney;
use topology::simplex::{graded_subsimplicies, SimplexExt};

use std::io::Write;

fn main() {
  let nnodes_dim = 20;
  let dim = 3;

  for simplex in graded_subsimplicies(dim).flatten() {
    let rank = simplex.dim();
    let simplex_string: String = simplex.iter().map(|i| i.to_string()).collect();
    let file = std::fs::File::create(format!("out/whitney{rank}_{simplex_string}.txt")).unwrap();
    let mut writer = std::io::BufWriter::new(file);

    for x in 0..nnodes_dim {
      for y in 0..(nnodes_dim - x) {
        for z in 0..(nnodes_dim - x - y) {
          let xf = x as f64 / (nnodes_dim - 1) as f64;
          let yf = y as f64 / (nnodes_dim - 1) as f64;
          let zf = z as f64 / (nnodes_dim - 1) as f64;
          let coord = na::DVector::from_row_slice(&[xf, yf, zf]);
          let form = ref_whitney(coord.as_view(), &simplex);

          write!(writer, "{xf:.4} {yf:.4} {zf:.4}").unwrap();
          for coeff in form.coeffs().iter() {
            write!(writer, " {coeff:.4}").unwrap();
          }
          writeln!(writer).unwrap();
        }
      }
    }
  }
}
