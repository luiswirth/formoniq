extern crate nalgebra as na;

use formoniq::fe::whitney::{ref_whitney, whitney_on_facet};
use geometry::coord::manifold::CoordSimplex;
use topology::simplex::{graded_subsimplicies, SimplexExt};

use std::io::Write;

fn main() {
  let nnodes_dim = 20;
  let dim = 2;

  let facet = na::dmatrix![
    1.0, 0.5, 0.0;
    0.0, 0.5, 0.5;
  ];
  let facet = CoordSimplex::new(facet);

  for simplex in graded_subsimplicies(dim).flatten() {
    let rank = simplex.dim();
    let simplex_string: String = simplex.iter().map(|i| i.to_string()).collect();
    let file = std::fs::File::create(format!("out/whitney{rank}_{simplex_string}.txt")).unwrap();
    let mut writer = std::io::BufWriter::new(file);

    for x in 0..nnodes_dim {
      for y in 0..(nnodes_dim - x) {
        let xf = x as f64 / (nnodes_dim - 1) as f64;
        let yf = y as f64 / (nnodes_dim - 1) as f64;
        let coord = facet.vertices.coord(0)
          + xf * facet.spanning_vectors().column(0)
          + yf * facet.spanning_vectors().column(1);
        let xf = coord[0];
        let yf = coord[1];
        //let form = ref_whitney(coord.as_view(), &simplex);
        let form = whitney_on_facet(coord.as_view(), &facet, &simplex);

        write!(writer, "{xf:.4} {yf:.4}").unwrap();
        for coeff in form.coeffs().iter() {
          write!(writer, " {coeff:.4}").unwrap();
        }
        writeln!(writer).unwrap();
      }
    }
  }
}
