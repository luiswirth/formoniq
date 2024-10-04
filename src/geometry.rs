use std::{
  cmp::{max, min},
  f64::consts::SQRT_2,
};

use num_integer::binomial;

use crate::{combinatorics::factorial, orientation::Orientation, Dim};

fn nedges(dim: Dim) -> usize {
  binomial(dim, 2)
}

#[derive(Debug, Clone)]
pub struct GeometrySimplex {
  /// Lengths of all edges in simplex
  /// Order of edges is lexicographical in vertex tuples.
  /// E.g. For a 3-simplex the edges are sorted as (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
  dim: Dim,
  edge_lengths: na::DVector<f64>,
}
impl GeometrySimplex {
  pub fn new(dim: Dim, edge_lengths: na::DVector<f64>) -> Self {
    assert!(edge_lengths.len() == nedges(dim));
    Self { dim, edge_lengths }
  }

  /// Constructs a reference simplex in `dim` dimensions.
  pub fn new_ref(dim: Dim) -> Self {
    let nedges = nedges(dim);
    let mut edge_lengths = na::DVector::zeros(nedges);
    for i in 0..dim {
      edge_lengths[i] = 1.0;
    }
    for i in dim..nedges {
      edge_lengths[i] = SQRT_2;
    }
    Self { dim, edge_lengths }
  }

  pub fn dim(&self) -> Dim {
    self.dim
  }
  pub fn nvertices(&self) -> usize {
    self.dim + 1
  }
  pub fn nedges(&self) -> usize {
    self.edge_lengths.len()
  }

  /// The determinate (signed volume) of the simplex.
  pub fn det(&self) -> f64 {
    (factorial(self.dim()) as f64).recip() * self.metric_tensor().determinant().sqrt()
  }

  /// The (unsigned) volume of the simplex.
  pub fn vol(&self) -> f64 {
    self.det().abs()
  }

  /// The orientation of the simplex.
  pub fn orientation(&self) -> Orientation {
    match self.det().is_sign_positive() {
      true => Orientation::Pos,
      false => Orientation::Neg,
    }
  }

  /// The diameter of the simplex.
  /// This is the maximum distance of two points inside the simplex.
  pub fn diameter(&self) -> f64 {
    self
      .edge_lengths
      .iter()
      .copied()
      .max_by(|a, b| a.partial_cmp(b).unwrap())
      .unwrap()
  }

  /// Returns the result of the Regge metric on the
  /// edge vectors (tangent vectors) i and j.
  /// This is the entry $G_(i j)$ of the metric tensor $G$.
  pub fn metric(&self, i: usize, j: usize) -> f64 {
    if i == j {
      self.edge_lengths[i].powi(2)
    } else {
      let ei = min(i, j);
      let ej = max(i, j);
      // TODO: make index computation more intuitive
      let eij = self.dim + ei * self.dim - ei * (ei - 1) / 2 + (ej - ei - 1);

      let l0i = self.edge_lengths[ei];
      let l0j = self.edge_lengths[ej];
      let lij = self.edge_lengths[eij];

      0.5 * l0i.powi(2) + l0j.powi(2) - lij.powi(2)
    }
  }

  pub fn metric_tensor(&self) -> na::DMatrix<f64> {
    let mut mat = na::DMatrix::zeros(self.dim, self.dim);
    for i in 0..self.dim {
      for j in i..self.dim {
        let v = self.metric(i, j);
        mat[(i, j)] = v;
        mat[(i, self.dim - 1 - j)] = v;
      }
    }
    mat
  }

  /// The shape regularity measure of the simplex.
  pub fn shape_reguarity_measure(&self) -> f64 {
    self.diameter().powi(self.dim() as i32) / self.vol()
  }

  /// Constant gradients of barycentric coordinate functions.
  pub fn barycentric_functions_grad(&self) -> na::DMatrix<f64> {
    todo!()
  }
}

pub fn ref_vol(dim: Dim) -> f64 {
  (factorial(dim) as f64).recip()
}

#[cfg(test)]
mod test {
  use super::*;

  #[test]
  fn ref_vol_test() {
    for d in 0..=8 {
      let simp = GeometrySimplex::new_ref(d);
      assert_eq!(simp.det(), ref_vol(d));
    }
  }

  #[test]
  fn reference_transform() {
    let refsimp = GeometrySimplex::new_ref(3);
    let simp = GeometrySimplex::new(
      3,
      na::DVector::from_column_slice(&[1.0, 1.0, 1.0, SQRT_2, SQRT_2, SQRT_2]),
    );
    assert_eq!(refsimp.edge_lengths, simp.edge_lengths);
  }
}
