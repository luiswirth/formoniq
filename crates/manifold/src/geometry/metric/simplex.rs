use super::EdgeIdx;
use crate::{geometry::coord::simplex::SimplexCoords, topology::simplex::nedges, Dim};

use common::{
  combo::{factorial, lex_rank},
  gramian::Gramian,
  linalg::nalgebra::{Matrix, Vector},
};

use itertools::Itertools;
use std::f64::consts::SQRT_2;

/// The edge lengths of a simplex.
///
/// Intrinsic geometry can be derived from this.
#[derive(Debug, Clone)]
pub struct SimplexLengths {
  /// Lexicographically ordered binom(dim+1,2) edge lengths
  lengths: Vector,
  /// Dimension of the simplex.
  dim: Dim,
}
impl SimplexLengths {
  pub fn new(lengths: Vector, dim: Dim) -> Self {
    assert_eq!(lengths.len(), nedges(dim), "Wrong number of edges.");
    let this = Self { lengths, dim };
    assert!(
      this.is_coordinate_realizable(),
      "Simplex must be coordiante realizable."
    );
    this
  }
  pub fn new_unchecked(lengths: Vector, dim: Dim) -> Self {
    if cfg!(debug_assertions) {
      Self::new(lengths, dim)
    } else {
      Self { lengths, dim }
    }
  }
  pub fn standard(dim: Dim) -> SimplexLengths {
    let nedges = nedges(dim);
    let lengths: Vec<f64> = (0..dim)
      .map(|_| 1.0)
      .chain((dim..nedges).map(|_| SQRT_2))
      .collect();

    Self::new_unchecked(lengths.into(), dim)
  }
  pub fn from_coords(coords: &SimplexCoords) -> Self {
    let dim = coords.dim_intrinsic();
    let lengths = coords.edges().map(|e| e.vol()).collect_vec().into();
    // SAFETY: Edge lengths stem from a realization already.
    Self::new_unchecked(lengths, dim)
  }

  pub fn dim(&self) -> Dim {
    self.dim
  }
  pub fn nvertices(&self) -> usize {
    self.dim() + 1
  }
  pub fn nedges(&self) -> usize {
    self.lengths.len()
  }
  pub fn length(&self, iedge: EdgeIdx) -> f64 {
    self[iedge]
  }

  /// The diameter of this cell.
  /// This is the maximum distance of two points inside the cell.
  pub fn diameter(&self) -> f64 {
    self
      .lengths
      .iter()
      .copied()
      .max_by(|a, b| a.partial_cmp(b).unwrap())
      .unwrap()
  }

  /// The shape regularity measure of this cell.
  pub fn shape_reguarity_measure(&self) -> f64 {
    self.diameter().powi(self.dim() as i32) / self.vol()
  }

  pub fn vector(&self) -> &Vector {
    &self.lengths
  }
  pub fn vector_mut(&mut self) -> &mut Vector {
    &mut self.lengths
  }
  pub fn into_vector(self) -> Vector {
    self.lengths
  }
  pub fn iter(
    &self,
  ) -> na::iter::MatrixIter<
    '_,
    f64,
    na::Dyn,
    na::Const<1>,
    na::VecStorage<f64, na::Dyn, na::Const<1>>,
  > {
    self.lengths.iter()
  }
}

impl std::ops::Index<EdgeIdx> for SimplexLengths {
  type Output = f64;
  fn index(&self, iedge: EdgeIdx) -> &Self::Output {
    &self.lengths[iedge]
  }
}

/// Distance Geometry
impl SimplexLengths {
  /// "Euclidean" distance matrix
  pub fn distance_matrix(&self) -> Matrix {
    let mut mat = Matrix::zeros(self.nvertices(), self.nvertices());

    let mut idx = 0;
    for i in 0..self.nvertices() {
      for j in (i + 1)..self.nvertices() {
        let dist_sqr = self.lengths[idx].powi(2);
        mat[(i, j)] = dist_sqr;
        mat[(j, i)] = dist_sqr;
        idx += 1;
      }
    }
    mat
  }
  pub fn cayley_menger_matrix(&self) -> Matrix {
    let mut mat = self.distance_matrix();
    mat = mat.insert_row(self.nvertices(), 1.0);
    mat = mat.insert_column(self.nvertices(), 1.0);
    mat[(self.nvertices(), self.nvertices())] = 0.0;
    mat
  }
  pub fn cayley_menger_det(&self) -> f64 {
    cayley_menger_factor(self.dim()) * self.cayley_menger_matrix().determinant()
  }
  pub fn is_coordinate_realizable(&self) -> bool {
    self.cayley_menger_det() >= 0.0
  }
  pub fn vol(&self) -> f64 {
    self.cayley_menger_det().sqrt()
  }
  pub fn is_degenerate(&self) -> bool {
    self.vol() <= 1e-12
  }
}
pub fn cayley_menger_factor(dim: Dim) -> f64 {
  (-1.0f64).powi(dim as i32 + 1) / factorial(dim).pow(2) as f64 / 2f64.powi(dim as i32)
}

impl SimplexLengths {
  pub fn from_regge_metric(metric: &Gramian) -> Self {
    let dim = metric.dim();
    let length = |i, j| {
      (metric.basis_inner(i, i) + metric.basis_inner(j, j) - 2.0 * metric.basis_inner(i, j)).sqrt()
    };

    let mut lengths = Vector::zeros(nedges(dim));
    let mut iedge = 0;
    for i in 0..dim {
      for j in i..dim {
        lengths[iedge] = length(i, j);
        iedge += 1;
      }
    }

    Self::new(lengths, dim)
  }

  pub fn to_regge_metric(&self) -> Gramian {
    let mut metric_tensor = Matrix::zeros(self.dim(), self.dim());
    for i in 0..self.dim() {
      metric_tensor[(i, i)] = self[i].powi(2);
    }
    for i in 0..self.dim() {
      for j in (i + 1)..self.dim() {
        let l0i = self[i];
        let l0j = self[j];

        let vi = i + 1;
        let vj = j + 1;
        let eij = lex_rank(&[vi, vj], self.nvertices());
        let lij = self[eij];

        let val = 0.5 * (l0i.powi(2) + l0j.powi(2) - lij.powi(2));

        metric_tensor[(i, j)] = val;
        metric_tensor[(j, i)] = val;
      }
    }
    Gramian::try_new(metric_tensor).expect("Edge Lengths must be coordinate realizable.")
  }
}
#[cfg(test)]
mod test {
  use super::*;
  use crate::geometry::coord::simplex::SimplexCoords;

  use approx::assert_relative_eq;

  #[test]
  fn ref_coord_vs_ref_lengths() {
    for dim in 0..=4 {
      let coords = SimplexCoords::standard(dim);
      let lengths = coords.to_lengths();
      assert_relative_eq!(lengths.vector(), SimplexLengths::standard(dim).vector());
      assert_relative_eq!(coords.vol(), lengths.vol());
    }
  }
}
