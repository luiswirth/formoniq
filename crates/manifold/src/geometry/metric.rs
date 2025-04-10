use super::{coord::local::SimplexCoords, refsimp_vol};
use crate::{
  topology::{
    complex::{
      handle::{SimplexHandle, SkeletonHandle},
      Complex,
    },
    simplex::nedges,
  },
  Dim,
};

use common::{
  combo::{factorial, lex_rank},
  gramian::Gramian,
};

use itertools::Itertools;
use rayon::iter::ParallelIterator;
use std::f64::consts::SQRT_2;

pub type EdgeIdx = usize;

#[derive(Debug, Clone)]
pub struct SimplexGeometry {
  lengths: SimplexLengths,
  metric: Gramian,
  inverse_metric: Gramian,
  // TODO: add multiform gramian
  //multiform_gramian: Gramian,
}
impl SimplexGeometry {
  pub fn new(lengths: SimplexLengths) -> Self {
    let metric = lengths.into_regge_metric();
    let inverse_metric = metric.clone().inverse();
    Self {
      lengths,
      metric,
      inverse_metric,
    }
  }
  pub fn standard(dim: Dim) -> Self {
    let lengths = SimplexLengths::standard(dim);
    let metric = Gramian::standard(dim);
    let inverse_metric = metric.clone();
    Self {
      lengths,
      metric,
      inverse_metric,
    }
  }

  pub fn lengths(&self) -> &SimplexLengths {
    &self.lengths
  }
  pub fn metric(&self) -> &Gramian {
    &self.metric
  }
  pub fn inverse_metric(&self) -> &Gramian {
    &self.inverse_metric
  }
  pub fn dim(&self) -> Dim {
    debug_assert_eq!(self.metric.dim(), self.lengths.dim());
    self.metric.dim()
  }
  pub fn nvertices(&self) -> usize {
    self.dim() + 1
  }

  /// The volume of this cell.
  pub fn vol(&self) -> f64 {
    refsimp_vol(self.dim()) * self.metric().det_sqrt()
  }

  /// The shape regularity measure of this cell.
  pub fn shape_reguarity_measure(&self) -> f64 {
    self.lengths.diameter().powi(self.dim() as i32) / self.vol()
  }
}

#[derive(Debug, Clone)]
pub struct MeshEdgeLengths {
  vector: na::DVector<f64>,
}
impl MeshEdgeLengths {
  pub fn new(vector: na::DVector<f64>, complex: &Complex) -> Self {
    Self::try_new(vector, complex).expect("Edge lengths are not coordinate realizable.")
  }
  pub fn try_new(vector: na::DVector<f64>, complex: &Complex) -> Option<Self> {
    let this = Self { vector };
    this
      .is_coordinate_realizable(complex.cells())
      .then_some(this)
  }
  pub fn new_unchecked(vector: na::DVector<f64>) -> Self {
    Self { vector }
  }
  pub fn standard(dim: usize) -> MeshEdgeLengths {
    let vector = SimplexLengths::standard(dim).into_vector();
    Self::new_unchecked(vector)
  }
  pub fn nedges(&self) -> usize {
    self.vector.len()
  }
  pub fn length(&self, iedge: EdgeIdx) -> f64 {
    self[iedge]
  }
  pub fn vector(&self) -> &na::DVector<f64> {
    &self.vector
  }
  pub fn vector_mut(&mut self) -> &mut na::DVector<f64> {
    &mut self.vector
  }
  pub fn into_vector(self) -> na::DVector<f64> {
    self.vector
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
    self.vector.iter()
  }

  /// The mesh width $h_max$, equal to the largest diameter of all cells.
  pub fn mesh_width_max(&self) -> f64 {
    self
      .iter()
      .max_by(|a, b| a.partial_cmp(b).unwrap())
      .copied()
      .unwrap()
  }

  /// By convexity the smallest length of a line inside a simplex is the length
  /// one of the edges.
  pub fn mesh_width_min(&self) -> f64 {
    self
      .iter()
      .min_by(|a, b| a.partial_cmp(b).unwrap())
      .copied()
      .unwrap()
  }

  /// The shape regularity measure $rho$ of the whole mesh, which is the largest
  /// shape regularity measure over all cells.
  pub fn shape_regularity_measure(&self, topology: &Complex) -> f64 {
    topology
      .cells()
      .iter()
      .map(|cell| self.simplex_geometry(cell).shape_reguarity_measure())
      .max_by(|a, b| a.partial_cmp(b).unwrap())
      .unwrap()
  }

  pub fn simplex_geometry(&self, simplex: SimplexHandle) -> SimplexGeometry {
    self.simplex_lengths(simplex).geometry()
  }

  pub fn simplex_lengths(&self, simplex: SimplexHandle) -> SimplexLengths {
    let lengths = simplex
      .edges()
      .map(|edge| self.length(edge.kidx()))
      .collect_vec()
      .into();
    // SAFETY: Already checked realizability.
    SimplexLengths::new_unchecked(lengths, simplex.dim())
  }

  pub fn is_coordinate_realizable(&self, skeleton: SkeletonHandle) -> bool {
    skeleton
      .par_iter()
      .all(|simp| self.simplex_lengths(simp).is_coordinate_realizable())
  }
}
impl std::ops::Index<EdgeIdx> for MeshEdgeLengths {
  type Output = f64;
  fn index(&self, iedge: EdgeIdx) -> &Self::Output {
    &self.vector[iedge]
  }
}

/// The edge lengths of a simplex.
///
/// Intrinsic geometry can be derived from this.
#[derive(Debug, Clone)]
pub struct SimplexLengths {
  /// Lexicographically ordered binom(dim+1,2) edge lengths
  lengths: na::DVector<f64>,
  /// Dimension of the simplex.
  dim: Dim,
}
impl SimplexLengths {
  pub fn new(lengths: na::DVector<f64>, dim: Dim) -> Self {
    assert_eq!(lengths.len(), nedges(dim), "Wrong number of edges.");
    let this = Self { lengths, dim };
    assert!(
      this.is_coordinate_realizable(),
      "Simplex must be coordiante realizable."
    );
    this
  }
  pub fn new_unchecked(lengths: na::DVector<f64>, dim: Dim) -> Self {
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

  pub fn geometry(self) -> SimplexGeometry {
    SimplexGeometry::new(self)
  }

  pub fn vector(&self) -> &na::DVector<f64> {
    &self.lengths
  }
  pub fn vector_mut(&mut self) -> &mut na::DVector<f64> {
    &mut self.lengths
  }
  pub fn into_vector(self) -> na::DVector<f64> {
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
  pub fn distance_matrix(&self) -> na::DMatrix<f64> {
    let mut mat = na::DMatrix::zeros(self.nvertices(), self.nvertices());

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
  pub fn cayley_menger_matrix(&self) -> na::DMatrix<f64> {
    let mut mat = self.distance_matrix();
    mat = mat.insert_row(self.nvertices(), 1.0);
    mat = mat.insert_column(self.nvertices(), 1.0);
    mat[(self.nvertices(), self.nvertices())] = 0.0;
    mat
  }
  pub fn cayley_menger_det(&self) -> f64 {
    cayley_menger_factor(self.dim()) * self.cayley_menger_matrix().determinant()
  }
  pub fn vol(&self) -> f64 {
    self.cayley_menger_det().sqrt()
  }
  pub fn is_degenerate(&self) -> bool {
    self.cayley_menger_det().abs() <= 1e-12
  }
  pub fn is_coordinate_realizable(&self) -> bool {
    self.cayley_menger_det() >= 0.0
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

    let mut lengths = na::DVector::zeros(nedges(dim));
    let mut iedge = 0;
    for i in 0..dim {
      for j in i..dim {
        lengths[iedge] = length(i, j);
        iedge += 1;
      }
    }

    Self::new(lengths, dim)
  }

  pub fn into_regge_metric(&self) -> Gramian {
    let mut metric_tensor = na::DMatrix::zeros(self.dim(), self.dim());
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

pub type MetricComplex = (Complex, MeshEdgeLengths);
pub fn standard_metric_complex(dim: Dim) -> MetricComplex {
  let topology = Complex::standard(dim);
  let lengths = MeshEdgeLengths::standard(dim);
  (topology, lengths)
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::geometry::coord::local::SimplexCoords;

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
