use super::EdgeIdx;
use crate::{topology::simplex::nedges, Dim};

use formoniq_linalg::nalgebra::{Matrix, Vector};
use gramian::{Gramian, RiemannianMetric};
use multiindex::{combinations, factorial, Combination};

use std::f64::consts::SQRT_2;

/// The edge lengths of a simplex.
///
/// Intrinsic geometry can be derived from this.
#[derive(Debug, Clone)]
pub struct SimplexLengths {
  /// The binom(dim+1,2) edge lengths, on the colexicographically ordered
  /// vertex pairs: the same order as
  /// [`Simplex::subsimps`](crate::topology::simplex::Simplex::subsimps) with dim 1.
  lengths: Vector,
  /// Dimension of the simplex.
  dim: Dim,
}

/// The edge index of a vertex pair: the colexicographic rank.
pub fn edge_index(vi: usize, vj: usize) -> EdgeIdx {
  Combination::from_increasing([vi, vj]).rank()
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
    // Edges containing the origin vertex are unit; all others connect two
    // standard basis vertices and have length sqrt(2).
    let lengths: Vec<f64> = combinations(dim + 1, 2)
      .map(|edge| if edge.contains(0) { 1.0 } else { SQRT_2 })
      .collect();

    Self::new_unchecked(lengths.into(), dim)
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
  pub fn shape_regularity_measure(&self) -> f64 {
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

    for (iedge, edge) in combinations(self.nvertices(), 2).enumerate() {
      let (vi, vj) = (edge.index_at(0), edge.index_at(1));
      let dist_sq = self.lengths[iedge].powi(2);
      mat[(vi, vj)] = dist_sq;
      mat[(vj, vi)] = dist_sq;
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
  /// Regge Calculus
  ///
  /// The spanning (basis) vector $e_i$ points from vertex $0$ to vertex
  /// $i + 1$, so edges from the origin are basis norms and
  /// $|v_j - v_i|^2 = g_(i-1,i-1) + g_(j-1,j-1) - 2 g_(i-1,j-1)$ otherwise.
  pub fn from_metric_tensor(metric: &Gramian) -> Self {
    let dim = metric.dim();

    let mut lengths = Vector::zeros(nedges(dim));
    for (iedge, edge) in combinations(dim + 1, 2).enumerate() {
      let (vi, vj) = (edge.index_at(0), edge.index_at(1));
      let length_sq = if vi == 0 {
        metric.basis_inner(vj - 1, vj - 1)
      } else {
        metric.basis_inner(vi - 1, vi - 1) + metric.basis_inner(vj - 1, vj - 1)
          - 2.0 * metric.basis_inner(vi - 1, vj - 1)
      };
      lengths[iedge] = length_sq.sqrt();
    }

    Self::new(lengths, dim)
  }

  /// The full Riemannian metric: the Gramian on tangent vectors together
  /// with its inverse on covectors.
  pub fn riemannian_metric(&self) -> RiemannianMetric {
    RiemannianMetric::new(self.to_metric_tensor())
  }

  /// Regge Calculus
  pub fn to_metric_tensor(&self) -> Gramian {
    let mut metric = Matrix::zeros(self.dim(), self.dim());
    for i in 0..self.dim() {
      metric[(i, i)] = self[edge_index(0, i + 1)].powi(2);
    }
    for i in 0..self.dim() {
      for j in (i + 1)..self.dim() {
        let l0i = self[edge_index(0, i + 1)];
        let l0j = self[edge_index(0, j + 1)];
        let lij = self[edge_index(i + 1, j + 1)];

        let val = 0.5 * (l0i.powi(2) + l0j.powi(2) - lij.powi(2));

        metric[(i, j)] = val;
        metric[(j, i)] = val;
      }
    }
    Gramian::new(metric)
  }
}
#[cfg(test)]
mod test {
  use super::*;

  use approx::assert_relative_eq;

  /// from_metric_tensor and to_metric_tensor are inverse.
  #[test]
  fn metric_tensor_roundtrip() {
    for dim in 1..=4 {
      let lengths = SimplexLengths::standard(dim);
      let roundtrip = SimplexLengths::from_metric_tensor(&lengths.to_metric_tensor());
      assert_relative_eq!(lengths.vector(), roundtrip.vector(), epsilon = 1e-12);
    }
  }
}
