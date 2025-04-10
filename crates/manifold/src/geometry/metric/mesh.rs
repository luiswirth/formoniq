use super::{simplex::SimplexLengths, EdgeIdx};
use crate::{
  topology::complex::{
    handle::{SimplexHandle, SkeletonHandle},
    Complex,
  },
  Dim,
};

use common::linalg::nalgebra::Vector;

use itertools::Itertools;
use rayon::iter::ParallelIterator;

/// The lengths of the edges of the mesh.
#[derive(Debug, Clone)]
pub struct MeshLengths {
  vector: Vector,
}
impl MeshLengths {
  pub fn new(vector: Vector, complex: &Complex) -> Self {
    Self::try_new(vector, complex).expect("Edge lengths are not coordinate realizable.")
  }
  pub fn try_new(vector: Vector, complex: &Complex) -> Option<Self> {
    let this = Self { vector };
    this
      .is_coordinate_realizable(complex.cells())
      .then_some(this)
  }
  pub fn new_unchecked(vector: Vector) -> Self {
    Self { vector }
  }
  pub fn standard(dim: usize) -> MeshLengths {
    let vector = SimplexLengths::standard(dim).into_vector();
    Self::new_unchecked(vector)
  }
  pub fn nedges(&self) -> usize {
    self.vector.len()
  }
  pub fn length(&self, iedge: EdgeIdx) -> f64 {
    self[iedge]
  }
  pub fn vector(&self) -> &Vector {
    &self.vector
  }
  pub fn vector_mut(&mut self) -> &mut Vector {
    &mut self.vector
  }
  pub fn into_vector(self) -> Vector {
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
      .map(|cell| self.simplex_lengths(cell).shape_reguarity_measure())
      .max_by(|a, b| a.partial_cmp(b).unwrap())
      .unwrap()
  }

  pub fn simplex_lengths(&self, simplex: SimplexHandle) -> SimplexLengths {
    let lengths = simplex
      .mesh_edges()
      .map(|edge| self.length(edge.kidx()))
      .collect_vec()
      .into();
    // SAFETY: Already checked realizability.
    SimplexLengths::new_unchecked(lengths, simplex.dim())
  }

  pub fn is_coordinate_realizable(&self, skeleton: SkeletonHandle) -> bool {
    skeleton
      .handle_par_iter()
      .all(|simp| self.simplex_lengths(simp).is_coordinate_realizable())
  }
}
impl std::ops::Index<EdgeIdx> for MeshLengths {
  type Output = f64;
  fn index(&self, iedge: EdgeIdx) -> &Self::Output {
    &self.vector[iedge]
  }
}

pub type MetricComplex = (Complex, MeshLengths);
pub fn standard_metric_complex(dim: Dim) -> MetricComplex {
  let topology = Complex::standard(dim);
  let lengths = MeshLengths::standard(dim);
  (topology, lengths)
}
