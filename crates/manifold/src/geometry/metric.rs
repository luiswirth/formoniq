use super::refsimp_vol;
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

use common::{combo::lex_rank, gramian::Gramian};

use itertools::Itertools;
use std::f64::consts::SQRT_2;

pub type EdgeIdx = usize;

#[derive(Debug, Clone)]
pub struct SimplexGeometry {
  edge_lengths: SimplexEdgeLengths,
  metric: Gramian,
  inverse_metric: Gramian,
  // TODO: add multiform gramian
  //multiform_gramian: Gramian,
}
impl SimplexGeometry {
  pub fn new(edge_lengths: SimplexEdgeLengths) -> Self {
    let metric = edge_lengths.compute_regge_metric();
    let inverse_metric = metric.clone().inverse();
    Self {
      edge_lengths,
      metric,
      inverse_metric,
    }
  }
  pub fn standard(dim: Dim) -> Self {
    let edge_lengths = SimplexEdgeLengths::standard(dim);
    let metric = Gramian::standard(dim);
    let inverse_metric = metric.clone();
    Self {
      edge_lengths,
      metric,
      inverse_metric,
    }
  }

  pub fn edge_lengths(&self) -> &SimplexEdgeLengths {
    &self.edge_lengths
  }
  pub fn metric(&self) -> &Gramian {
    &self.metric
  }
  pub fn inverse_metric(&self) -> &Gramian {
    &self.inverse_metric
  }
  pub fn dim(&self) -> Dim {
    debug_assert_eq!(self.metric.dim(), self.edge_lengths.dim());
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
    self.edge_lengths.diameter().powi(self.dim() as i32) / self.vol()
  }
}

#[derive(Debug, Clone)]
pub struct MeshEdgeLengths {
  vector: na::DVector<f64>,
}
impl MeshEdgeLengths {
  pub fn new(vector: na::DVector<f64>, complex: &Complex) -> Self {
    Self::try_new(vector, complex).expect("Edge Lengths are not coordinate realizable.")
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
    let vector = SimplexEdgeLengths::standard(dim).into_vector();
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
      .handle_iter()
      .map(|cell| self.simplex_geometry(cell).shape_reguarity_measure())
      .max_by(|a, b| a.partial_cmp(b).unwrap())
      .unwrap()
  }

  pub fn simplex_geometry(&self, simplex: SimplexHandle) -> SimplexGeometry {
    self.simplex_edge_lengths(simplex).geometry()
  }

  pub fn simplex_edge_lengths(&self, simplex: SimplexHandle) -> SimplexEdgeLengths {
    let lengths = simplex
      .edges()
      .map(|edge| self.length(edge.kidx()))
      .collect_vec()
      .into();
    SimplexEdgeLengths::new(lengths, simplex.dim())
  }

  pub fn is_coordinate_realizable(&self, skeleton: SkeletonHandle) -> bool {
    skeleton
      .handle_iter()
      .all(|simp| self.simplex_edge_lengths(simp).is_coordinate_realizable())
  }
}
impl std::ops::Index<EdgeIdx> for MeshEdgeLengths {
  type Output = f64;
  fn index(&self, iedge: EdgeIdx) -> &Self::Output {
    &self.vector[iedge]
  }
}

#[derive(Debug, Clone)]
pub struct SimplexEdgeLengths {
  /// Lexicographically ordered binom(n+1,2) edge lengths
  edge_lengths: na::DVector<f64>,
  dim: Dim,
}
impl SimplexEdgeLengths {
  pub fn new(edge_lengths: na::DVector<f64>, dim: Dim) -> Self {
    Self { edge_lengths, dim }
  }
  pub fn standard(dim: Dim) -> SimplexEdgeLengths {
    let nedges = nedges(dim);
    let edge_lengths: Vec<f64> = (0..dim)
      .map(|_| 1.0)
      .chain((dim..nedges).map(|_| SQRT_2))
      .collect();
    Self::new(edge_lengths.into(), dim)
  }
  pub fn from_metric(metric: &Gramian) -> Self {
    let dim = metric.dim();

    let edge_len = |i, j| {
      (metric.basis_inner(i, i) + metric.basis_inner(j, j) - 2.0 * metric.basis_inner(i, j)).sqrt()
    };

    let mut edge_lengths = na::DVector::zeros(nedges(dim));
    let mut iedge = 0;
    for i in 0..dim {
      for j in i..dim {
        edge_lengths[iedge] = edge_len(i, j);
        iedge += 1;
      }
    }

    Self::new(edge_lengths, dim)
  }

  pub fn nedges(&self) -> usize {
    self.edge_lengths.len()
  }
  pub fn length(&self, iedge: EdgeIdx) -> f64 {
    self[iedge]
  }

  pub fn dim(&self) -> Dim {
    self.dim
  }
  pub fn nvertices(&self) -> usize {
    self.dim() + 1
  }

  /// The diameter of this cell.
  /// This is the maximum distance of two points inside the cell.
  pub fn diameter(&self) -> f64 {
    self
      .edge_lengths
      .iter()
      .copied()
      .max_by(|a, b| a.partial_cmp(b).unwrap())
      .unwrap()
  }

  pub fn geometry(self) -> SimplexGeometry {
    SimplexGeometry::new(self)
  }

  pub fn vector(&self) -> &na::DVector<f64> {
    &self.edge_lengths
  }
  pub fn vector_mut(&mut self) -> &mut na::DVector<f64> {
    &mut self.edge_lengths
  }
  pub fn into_vector(self) -> na::DVector<f64> {
    self.edge_lengths
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
    self.edge_lengths.iter()
  }

  /// "Euclidean" distance matrix
  pub fn distance_matrix(&self) -> na::DMatrix<f64> {
    let mut mat = na::DMatrix::zeros(self.dim(), self.dim());

    let mut idx = 0;
    for i in 0..self.nvertices() {
      for j in (i + 1)..self.nvertices() {
        let dist_sqr = self.edge_lengths[idx].powi(2);
        mat[(i, j)] = dist_sqr;
        mat[(j, i)] = dist_sqr;
        idx += 1;
      }
    }
    mat
  }

  pub fn cayley_menger_matrix(&self) -> na::DMatrix<f64> {
    let mat = self.distance_matrix();
    let mat = mat.insert_row(self.dim(), 1.0);
    let mut mat = mat.insert_column(self.dim(), 1.0);
    mat[(self.dim(), self.dim())] = 0.0;
    mat
  }
  pub fn cayley_menger_det_unscaled(&self) -> f64 {
    self.cayley_menger_matrix().determinant()
  }
  pub fn cayley_menger_det(&self) -> f64 {
    cayley_menger_factor(self.dim()) * self.cayley_menger_det_unscaled()
  }
  pub fn vol(&self) -> f64 {
    self.cayley_menger_det().sqrt()
  }
  pub fn is_degenerate(&self) -> bool {
    self.cayley_menger_det_unscaled().abs() <= 1e-12
  }
  pub fn is_coordinate_realizable(&self) -> bool {
    self.cayley_menger_det_unscaled() >= 0.0
  }

  /// Builds regge metric tensor from edge lenghts of simplex.
  pub fn compute_regge_metric(&self) -> Gramian {
    let dim = self.dim();
    let nvertices = dim + 1;
    let mut metric_tensor = na::DMatrix::zeros(dim, dim);
    for i in 0..dim {
      metric_tensor[(i, i)] = self[i].powi(2);
    }
    for i in 0..dim {
      for j in (i + 1)..dim {
        let l0i = self[i];
        let l0j = self[j];

        let vi = i + 1;
        let vj = j + 1;
        let eij = lex_rank(&[vi, vj], nvertices);
        let lij = self[eij];

        let val = 0.5 * (l0i.powi(2) + l0j.powi(2) - lij.powi(2));

        metric_tensor[(i, j)] = val;
        metric_tensor[(j, i)] = val;
      }
    }
    Gramian::try_new(metric_tensor).expect("Edge Lengths must be coordinate realizable.")
  }
}

pub fn cayley_menger_factor(dim: Dim) -> f64 {
  -1.0f64.powi(dim as i32 + 1) * refsimp_vol(dim).powi(2) / 2f64.powi(dim as i32)
}

impl std::ops::Index<EdgeIdx> for SimplexEdgeLengths {
  type Output = f64;
  fn index(&self, iedge: EdgeIdx) -> &Self::Output {
    &self.edge_lengths[iedge]
  }
}

pub type MetricComplex = (Complex, MeshEdgeLengths);
pub fn standard_metric_complex(dim: Dim) -> MetricComplex {
  let topology = Complex::standard(dim);
  let edge_lengths = MeshEdgeLengths::standard(dim);
  (topology, edge_lengths)
}
