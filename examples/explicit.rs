#![allow(clippy::useless_vec)]

use formoniq::combinatorics::Orientation;

extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

fn main() {
  let vertices = vec![
    na::DVector::from_column_slice(&[0.0, 0.0, 0.0]),
    na::DVector::from_column_slice(&[1.0, 0.0, 0.0]),
    na::DVector::from_column_slice(&[0.0, 1.0, 0.0]),
    na::DVector::from_column_slice(&[1.0, 1.0, 0.0]),
    na::DVector::from_column_slice(&[0.0, 0.0, 1.0]),
    na::DVector::from_column_slice(&[1.0, 0.0, 1.0]),
    na::DVector::from_column_slice(&[0.0, 1.0, 1.0]),
    na::DVector::from_column_slice(&[1.0, 1.0, 1.0]),
  ];

  let tets = vec![
    vec![0, 1, 3, 7],
    vec![0, 1, 5, 7],
    vec![0, 2, 3, 7],
    vec![0, 2, 6, 7],
    vec![0, 4, 5, 7],
    vec![0, 4, 6, 7],
  ];
  let tet_vol = 1.0 / 6.0;

  let mut galmat = na::DMatrix::<f64>::zeros(vertices.len(), vertices.len());
  for (itet, tet_ivertices) in tets.iter().enumerate() {
    let tet_orientation = Orientation::from_permutation_parity(itet);

    let tet_vertices: Vec<_> = tet_ivertices.iter().map(|&i| vertices[i].clone()).collect();

    let ns: Vec<na::DVector<f64>> = (0..4)
      .map(|i| {
        let mut face = tet_vertices.clone();
        face.remove(i);

        let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
        let b0 = &face[1] - &face[0];
        let b1 = &face[2] - &face[0];
        sign * b0.cross(&b1)
      })
      .collect();

    let mut elmat = na::DMatrix::zeros(4, 4);
    for i in 0..4 {
      for j in 0..4 {
        let ni = &ns[i];
        let nj = &ns[j];
        elmat[(i, j)] = tet_orientation.as_f64() * ni.dot(nj);
      }
    }
    elmat *= 1.0 / (36.0 * tet_vol);
    println!("Elmat #{itet}");
    println!("{elmat:.3}");

    for (ilocal, iglobal) in tet_ivertices.iter().copied().enumerate() {
      for (jlocal, jglobal) in tet_ivertices.iter().copied().enumerate() {
        galmat[(iglobal, jglobal)] += elmat[(ilocal, jlocal)];
      }
    }
  }

  println!("Galmat");
  println!("{galmat:.3}");
}
