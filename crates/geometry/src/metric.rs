pub mod manifold;

use common::linalg::DMatrixExt;
use index_algebra::{factorial, variants::SetOrder, IndexSet};
use itertools::Itertools;
use topology::{
  complex::TopologyComplex,
  simplex::{nsubsimplicies, Simplex},
  Dim,
};

use std::f64::consts::SQRT_2;

pub type EdgeIdx = usize;

pub fn ref_vol(dim: Dim) -> f64 {
  (factorial(dim) as f64).recip()
}

#[derive(Debug, Clone)]
pub struct SimplexGeometry {
  edge_lengths: SimplexEdgeLengths,
  metric: RiemannianMetric,
}
impl SimplexGeometry {
  pub fn new(edge_lengths: SimplexEdgeLengths) -> Self {
    let metric = compute_regge_metric(&edge_lengths, edge_lengths.dim());
    Self {
      edge_lengths,
      metric,
    }
  }
  pub fn standard(dim: Dim) -> Self {
    let edge_lengths = SimplexEdgeLengths::standard(dim);
    let metric = RiemannianMetric::standard(dim);
    Self {
      edge_lengths,
      metric,
    }
  }

  pub fn edge_lengths(&self) -> &SimplexEdgeLengths {
    &self.edge_lengths
  }
  pub fn metric(&self) -> &RiemannianMetric {
    &self.metric
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
    ref_vol(self.dim()) * self.metric().det_sqrt()
  }

  /// The shape regularity measure of this cell.
  pub fn shape_reguarity_measure(&self) -> f64 {
    self.edge_lengths.diameter().powi(self.dim() as i32) / self.vol()
  }
}

#[derive(Debug, Clone)]
pub struct RiemannianMetric {
  metric_tensor: na::DMatrix<f64>,
  inverse_metric_tensor: na::DMatrix<f64>,
}
impl RiemannianMetric {
  pub fn new(metric_tensor: na::DMatrix<f64>) -> Self {
    let n = metric_tensor.nrows();
    // WARN: Numerically Unstable. TODO: can we avoid this?
    let inverse_metric_tensor = metric_tensor
      .clone()
      .cholesky()
      .unwrap()
      .solve(&na::DMatrix::identity(n, n));
    Self {
      metric_tensor,
      inverse_metric_tensor,
    }
  }

  pub fn from_tangent_basis(basis: na::DMatrix<f64>) -> Self {
    let metric_tensor = basis.gramian();
    Self::new(metric_tensor)
  }

  /// Orthonormal euclidean metric.
  pub fn standard(dim: Dim) -> Self {
    let identity = na::DMatrix::identity(dim, dim);
    let metric_tensor = identity.clone();
    let inverse_metric_tensor = identity;
    Self {
      metric_tensor,
      inverse_metric_tensor,
    }
  }

  pub fn metric_tensor(&self) -> &na::DMatrix<f64> {
    &self.metric_tensor
  }
  pub fn inverse_metric_tensor(&self) -> &na::DMatrix<f64> {
    &self.inverse_metric_tensor
  }

  pub fn inner(&self, i: usize, j: usize) -> f64 {
    self.metric_tensor[(i, j)]
  }
  pub fn length_sqr(&self, i: usize) -> f64 {
    self.inner(i, i)
  }
  pub fn length(&self, i: usize) -> f64 {
    self.length_sqr(i).sqrt()
  }
  pub fn angle_cos(&self, i: usize, j: usize) -> f64 {
    self.inner(i, j) / self.length(i) / self.length(j)
  }
  pub fn angle(&self, i: usize, j: usize) -> f64 {
    self.angle_cos(i, j).acos()
  }

  pub fn dim(&self) -> Dim {
    self.metric_tensor.nrows()
  }
  pub fn det(&self) -> f64 {
    self.metric_tensor.determinant()
  }
  pub fn det_sqrt(&self) -> f64 {
    self.det().sqrt()
  }

  pub fn simplex_edge_length_sqr(&self, i: usize, j: usize) -> f64 {
    self.inner(i, i) + self.inner(j, j) - 2.0 * self.inner(i, j)
  }
  pub fn simplex_edge_length(&self, i: usize, j: usize) -> f64 {
    self.simplex_edge_length_sqr(i, j).sqrt()
  }

  /// Orthonormal (w.r.t. metric) vectors expressed using old basis.
  ///
  /// Orthonormalizes metric tensor $I = B^T G B = B^T V^T V B = (V B)^T (V B)$
  pub fn orthonormal_basis(&self) -> na::DMatrix<f64> {
    let na::SymmetricEigen {
      eigenvalues,
      mut eigenvectors,
    } = self.metric_tensor.clone().symmetric_eigen();
    for (eigenvalue, mut eigenvector) in eigenvalues.iter().zip(eigenvectors.column_iter_mut()) {
      eigenvector /= eigenvalue.sqrt();
    }
    eigenvectors
  }

  pub fn orthormal_change_of_basis(&self) -> na::DMatrix<f64> {
    self.orthonormal_basis().try_inverse().unwrap()
  }

  /// Gram matrix on tangent vector standard basis.
  pub fn vector_gramian(&self) -> &na::DMatrix<f64> {
    &self.metric_tensor
  }

  pub fn vector_inner_product(&self, v: &na::DVector<f64>, w: &na::DVector<f64>) -> f64 {
    (v.transpose() * self.vector_gramian() * w).x
  }

  pub fn vector_inner_product_mat(
    &self,
    v: &na::DMatrix<f64>,
    w: &na::DMatrix<f64>,
  ) -> na::DMatrix<f64> {
    v.transpose() * self.vector_gramian() * w
  }
  pub fn vector_norm_sqr(&self, v: &na::DMatrix<f64>) -> na::DMatrix<f64> {
    self.vector_inner_product_mat(v, v)
  }

  /// Gram matrix on tangent covector standard basis.
  pub fn covector_gramian(&self) -> &na::DMatrix<f64> {
    &self.inverse_metric_tensor
  }
  pub fn covector_inner_product(
    &self,
    v: &na::DMatrix<f64>,
    w: &na::DMatrix<f64>,
  ) -> na::DMatrix<f64> {
    v.transpose() * self.covector_gramian() * w
  }
  pub fn covector_norm_sqr(&self, v: &na::DMatrix<f64>) -> na::DMatrix<f64> {
    self.covector_inner_product(v, v)
  }
}

#[derive(Debug, Clone)]
pub struct MeshEdgeLengths {
  vector: na::DVector<f64>,
}
impl MeshEdgeLengths {
  pub fn new(vector: na::DVector<f64>) -> Self {
    Self { vector }
  }
  pub fn standard(dim: usize) -> MeshEdgeLengths {
    let vector = SimplexEdgeLengths::standard(dim).into_vector();
    Self::new(vector)
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
  pub fn shape_regularity_measure(&self, topology: &TopologyComplex) -> f64 {
    topology
      .facets()
      .handle_iter()
      .map(|facet| {
        self
          .simplex_geometry(facet.simplex_set())
          .shape_reguarity_measure()
      })
      .max_by(|a, b| a.partial_cmp(b).unwrap())
      .unwrap()
  }

  pub fn simplex_geometry<O: SetOrder>(&self, simplex: &Simplex<O>) -> SimplexGeometry {
    self.simplex_edge_lengths(simplex).geometry()
  }

  pub fn simplex_edge_lengths<O: SetOrder>(&self, simplex: &Simplex<O>) -> SimplexEdgeLengths {
    let lengths = simplex
      .vertices
      .iter()
      .map(|edge| self.length(edge))
      .collect_vec()
      .into();
    SimplexEdgeLengths::new(lengths, simplex.dim())
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
  edge_lengths: na::DVector<f64>,
  dim: Dim,
}
impl SimplexEdgeLengths {
  pub fn new(edge_lengths: na::DVector<f64>, dim: Dim) -> Self {
    Self { edge_lengths, dim }
  }
  pub fn standard(dim: Dim) -> SimplexEdgeLengths {
    let nedges = nsubsimplicies(dim, 1);
    let edge_lengths: Vec<f64> = (0..dim)
      .map(|_| 1.0)
      .chain((dim..nedges).map(|_| SQRT_2))
      .collect();
    Self::new(edge_lengths.into(), dim)
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
}

impl std::ops::Index<EdgeIdx> for SimplexEdgeLengths {
  type Output = f64;
  fn index(&self, iedge: EdgeIdx) -> &Self::Output {
    &self.edge_lengths[iedge]
  }
}

/// Builds regge metric tensor from edge lenghts of simplex.
///
/// On the simplicial manifold the edge vectors are the tangent vectors.
pub fn compute_regge_metric(edge_lengths: &SimplexEdgeLengths, dim: Dim) -> RiemannianMetric {
  let nvertices = dim + 1;
  let mut metric_tensor = na::DMatrix::zeros(dim, dim);
  for i in 0..dim {
    metric_tensor[(i, i)] = edge_lengths[i].powi(2);
  }
  for i in 0..dim {
    for j in (i + 1)..dim {
      let l0i = edge_lengths[i];
      let l0j = edge_lengths[j];

      let vi = i + 1;
      let vj = j + 1;
      // TODO: can we compute this more directly?
      let eij = IndexSet::from([vi, vj]).assume_sorted().lex_rank(nvertices);
      let lij = edge_lengths[eij];

      let val = 0.5 * (l0i.powi(2) + l0j.powi(2) - lij.powi(2));

      metric_tensor[(i, j)] = val;
      metric_tensor[(j, i)] = val;
    }
  }
  RiemannianMetric::new(metric_tensor)
}

pub type MetricComplex = (TopologyComplex, MeshEdgeLengths);
pub fn standard_metric_complex(dim: Dim) -> MetricComplex {
  let topology = TopologyComplex::standard(dim);
  let edge_lengths = MeshEdgeLengths::standard(dim);
  (topology, edge_lengths)
}

#[cfg(test)]
mod test {
  use super::*;

  #[test]
  fn orthonormal_basis() {
    let dim = 2;
    let tangent_vectors = na::dmatrix![
      1.0, 0.0;
      1.0, 1.0;
    ];
    let metric = RiemannianMetric::from_tangent_basis(tangent_vectors.clone());
    let orthonormal_basis = metric.orthonormal_basis();
    assert!(
      (orthonormal_basis.transpose() * &metric.metric_tensor * &orthonormal_basis
        - na::DMatrix::identity(dim, dim))
      .norm()
        <= 1e-12
    );

    let orthonormal_tangent_vectors = tangent_vectors * &orthonormal_basis;
    let orthogonal_metric = RiemannianMetric::from_tangent_basis(orthonormal_tangent_vectors);
    assert!((orthogonal_metric.metric_tensor() - na::DMatrix::identity(dim, dim)).norm() <= 1e-12);

    let vector_a = na::dvector![0.2, 5.3];
    let vector_b = na::dvector![-1.3, 2.8];
    let inner0 = metric.vector_inner_product(&vector_a, &vector_b);
    let cob = metric.orthormal_change_of_basis();
    let inner1 = (&cob * vector_a).dot(&(&cob * vector_b));
    assert_eq!(inner0, inner1);
  }
}
