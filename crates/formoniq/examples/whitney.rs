extern crate nalgebra as na;

use exterior::dense::ExteriorField;
use formoniq::whitney::WhitneyForm;
use geometry::coord::manifold::SimplexCoords;
use topology::simplex::graded_subsimplicies;

use std::io::Write;

fn main() {
  let nnodes_dim = 20;

  let cell = na::dmatrix![
    1.0, 0.5, 0.0;
    0.0, 0.5, 0.5;
  ];
  let cell = SimplexCoords::new(cell);
  let dim = cell.dim_intrinsic();

  for simplex in graded_subsimplicies(dim).flatten() {
    let whitney_form = WhitneyForm::new(cell.clone(), simplex.clone());

    let grade = simplex.dim();
    let simplex_string: String = simplex.vertices.iter().map(|i| i.to_string()).collect();
    let file = std::fs::File::create(format!("out/whitney{grade}_{simplex_string}.txt")).unwrap();
    let mut writer = std::io::BufWriter::new(file);

    for x in 0..nnodes_dim {
      for y in 0..(nnodes_dim - x) {
        let xf = x as f64 / (nnodes_dim - 1) as f64;
        let yf = y as f64 / (nnodes_dim - 1) as f64;
        let coord = cell.vertices.coord(0)
          + xf * cell.spanning_vectors().column(0)
          + yf * cell.spanning_vectors().column(1);
        let xf = coord[0];
        let yf = coord[1];

        let form = whitney_form.at_point(&coord);

        write!(writer, "{xf:.4} {yf:.4}").unwrap();
        for coeff in form.coeffs().iter() {
          write!(writer, " {coeff:.4}").unwrap();
        }
        writeln!(writer).unwrap();
      }
    }
  }
}
