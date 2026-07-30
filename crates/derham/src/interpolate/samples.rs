//! A family of local shape functions, sampled at a set of points.

use super::form::WhitneyLsf;

use {
  multialgebra::{Dim, ExteriorGrade, Tensor},
  simplicial::atlas::Bary,
};

/// The values of a DOF-indexed family of shape forms at a sequence of points,
/// indexed `[node][dof]`.
///
/// Reference data: the shape functions and the sample points depend on the cell
/// dimension, the grade and the rule alone, so the table is built once and read
/// on every cell. Not an evaluation at the degrees of freedom: a node is
/// where a quadrature asks, a DOF is which shape function answers.
///
/// A family constant over the nodes, such as [`WhitneyLsf::dif`], is the
/// degenerate member rather than a special case, which is why an integrand
/// never has to reach a DOF index to find a precomputed constant.
#[derive(Debug, Clone)]
pub struct LsfSamples {
  grade: ExteriorGrade,
  /// `[node][dof]`.
  values: Vec<Vec<Tensor>>,
}

impl LsfSamples {
  /// The Whitney shape functions of a grade, at the given points.
  pub fn whitney(dim: impl Into<Dim>, grade: impl Into<ExteriorGrade>, nodes: &[Bary]) -> Self {
    let (dim, grade) = (dim.into(), grade.into());
    let lsfs: Vec<_> = WhitneyLsf::basis(dim, grade).collect();
    let values = nodes
      .iter()
      .map(|bary| lsfs.iter().map(|lsf| lsf.at_bary(bary)).collect())
      .collect();
    Self { grade, values }
  }

  /// The differentials of the Whitney shape functions of a grade, which are
  /// constant on the cell and so take the same value at every node.
  ///
  /// The resulting family has grade $k+1$.
  pub fn whitney_dif(dim: impl Into<Dim>, grade: impl Into<ExteriorGrade>, nnodes: usize) -> Self {
    let (dim, grade) = (dim.into(), grade.into());
    let difs: Vec<_> = WhitneyLsf::basis(dim, grade).map(|lsf| lsf.dif()).collect();
    Self {
      grade: grade + 1,
      values: vec![difs; nnodes],
    }
  }

  pub fn grade(&self) -> ExteriorGrade {
    self.grade
  }
  pub fn ndofs(&self) -> usize {
    self.values.first().map_or(0, Vec::len)
  }
  pub fn nnodes(&self) -> usize {
    self.values.len()
  }

  /// The values of every shape function at one node.
  pub fn at_node(&self, inode: usize) -> &[Tensor] {
    &self.values[inode]
  }
}
