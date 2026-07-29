//! A family of local shape functions, sampled at a set of points.

use super::form::WhitneyLsf;

use {
  multialgebra::{Dim, ExteriorGrade, Tensor},
  simplicial::{atlas::Bary, topology::simplex::unit_subsimps},
};

/// The values of a DOF-indexed family of shape forms at a sequence of points,
/// indexed `[node][dof]`.
///
/// Reference data: the shape functions and the sample points depend on the cell
/// dimension, the grade and the rule alone, so the table is built once and read
/// on every cell. Not an evaluation *at* the degrees of freedom -- a node is
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
    let lsfs: Vec<_> = unit_subsimps(dim, grade)
      .map(|dof_simp| WhitneyLsf::unit(dim, dof_simp))
      .collect();
    let values = nodes
      .iter()
      .map(|bary| lsfs.iter().map(|lsf| lsf.at_bary(bary)).collect())
      .collect();
    Self { grade, values }
  }

  /// The trimmed shape functions $P^-_r Lambda^k$ of a geometric
  /// decomposition, at the given points, in the decomposition's own dof order.
  ///
  /// The general family the Whitney one is the $r = 1$ case of.
  pub fn trimmed(
    decomposition: &crate::decomposition::GeometricDecomposition,
    nodes: &[Bary],
  ) -> Self {
    let basis = decomposition.local_basis();
    let values = nodes
      .iter()
      .map(|bary| basis.iter().map(|(_, form)| form.at_bary(bary)).collect())
      .collect();
    Self {
      grade: decomposition.grade(),
      values,
    }
  }

  /// The differentials of the Whitney shape functions of a grade, which are
  /// constant on the cell and so take the same value at every node.
  ///
  /// The resulting family has grade $k+1$.
  pub fn whitney_dif(dim: impl Into<Dim>, grade: impl Into<ExteriorGrade>, nnodes: usize) -> Self {
    let (dim, grade) = (dim.into(), grade.into());
    let difs: Vec<_> = unit_subsimps(dim, grade)
      .map(|dof_simp| WhitneyLsf::unit(dim, dof_simp).dif())
      .collect();
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
