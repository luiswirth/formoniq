use super::EdgeIdx;
use crate::{Dim, topology::simplex::nedges};

use crate::linalg::{Matrix, Vector};
use gramian::{CausalType, Gramian, Metric};
use multiindex::{Combination, combinations, factorial};

/// The signed squared edge lengths of a simplex: Regge calculus, on any
/// metric signature.
///
/// The squared length $s_(i j) = norm(v_j - v_i)^2_g$ is the Regge primitive
/// -- signed, exactly like [`Gramian::norm_sq`]: positive on a spacelike
/// edge, zero on a null one, negative on a timelike one. Regge invented the
/// calculus for Lorentzian spacetimes ("general relativity without
/// coordinates"), and the squared length is what makes that work: the metric
/// tensor is a polarization identity in the $s_(i j)$, rational and
/// signature-blind, while an unsquared length would lose the causal sign
/// under the square root. Riemannian geometry is the all-positive,
/// Euclidean-realizable corner, not a separate representation.
#[derive(Debug, Clone)]
pub struct SimplexLengthsSq {
  /// The binom(dim+1,2) signed squared edge lengths, on the
  /// colexicographically ordered vertex pairs: the same order as
  /// [`Simplex::subsimps`](crate::topology::simplex::Simplex::subsimps) with dim 1.
  lengths_sq: Vector,
  /// Dimension of the simplex.
  dim: Dim,
}

/// The edge index of a vertex pair: the colexicographic rank.
pub fn edge_index(vi: usize, vj: usize) -> EdgeIdx {
  Combination::from_increasing([vi, vj]).rank()
}
impl SimplexLengthsSq {
  /// The invariant is non-degeneracy of the induced metric tensor -- the
  /// squared lengths must describe a simplex of *some* signature, Euclidean
  /// realizability ([`Self::is_coordinate_realizable`]) being the Riemannian
  /// special case, not the requirement.
  pub fn new(lengths_sq: Vector, dim: Dim) -> Self {
    assert_eq!(lengths_sq.len(), nedges(dim), "Wrong number of edges.");
    let this = Self { lengths_sq, dim };
    assert!(
      !this.is_degenerate(),
      "Simplex metric must be non-degenerate."
    );
    this
  }
  pub fn new_unchecked(lengths_sq: Vector, dim: Dim) -> Self {
    if cfg!(debug_assertions) {
      Self::new(lengths_sq, dim)
    } else {
      Self { lengths_sq, dim }
    }
  }
  /// The reference simplex: edges at the origin vertex are unit, all others
  /// connect two standard basis vertices with squared length $2$.
  pub fn standard(dim: Dim) -> SimplexLengthsSq {
    let lengths_sq: Vec<f64> = combinations(dim + 1, 2)
      .map(|edge| if edge.contains(0) { 1.0 } else { 2.0 })
      .collect();

    Self::new_unchecked(lengths_sq.into(), dim)
  }
  pub fn dim(&self) -> Dim {
    self.dim
  }
  pub fn nvertices(&self) -> usize {
    self.dim() + 1
  }
  pub fn nedges(&self) -> usize {
    self.lengths_sq.len()
  }
  /// The signed squared length of an edge: the Regge primitive, its sign the
  /// causal character.
  pub fn length_sq(&self, iedge: EdgeIdx) -> f64 {
    self[iedge]
  }
  /// The magnitude $sqrt(abs(s))$ of an edge. On an indefinite metric this is
  /// meaningful only together with [`Self::causal_type`]; it is never NaN.
  pub fn length(&self, iedge: EdgeIdx) -> f64 {
    self[iedge].abs().sqrt()
  }
  /// The causal character of an edge: the sign of its squared length.
  pub fn causal_type(&self, iedge: EdgeIdx) -> CausalType {
    CausalType::from_norm_sq(self[iedge])
  }

  /// The diameter of this cell: the largest edge magnitude, which by
  /// convexity bounds the distance of any two points inside. A metric
  /// quantity of the Riemannian case; on an indefinite metric it is a mesh
  /// scale, not a distance.
  pub fn diameter(&self) -> f64 {
    self
      .lengths_sq
      .iter()
      .map(|s| s.abs())
      .max_by(|a, b| a.partial_cmp(b).unwrap())
      .unwrap()
      .sqrt()
  }

  /// The shape regularity measure of this cell.
  pub fn shape_regularity(&self) -> f64 {
    self.diameter().powi(self.dim() as i32) / self.vol()
  }

  pub fn vector(&self) -> &Vector {
    &self.lengths_sq
  }
  pub fn vector_mut(&mut self) -> &mut Vector {
    &mut self.lengths_sq
  }
  pub fn into_vector(self) -> Vector {
    self.lengths_sq
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
    self.lengths_sq.iter()
  }
}

impl std::ops::Index<EdgeIdx> for SimplexLengthsSq {
  type Output = f64;
  fn index(&self, iedge: EdgeIdx) -> &Self::Output {
    &self.lengths_sq[iedge]
  }
}

/// Distance Geometry
impl SimplexLengthsSq {
  /// The matrix of signed squared distances between the vertices: exactly the
  /// stored Regge data, symmetrized.
  pub fn distance_matrix(&self) -> Matrix {
    let mut mat = Matrix::zeros(self.nvertices(), self.nvertices());

    for (iedge, edge) in combinations(self.nvertices(), 2).enumerate() {
      let (vi, vj) = (edge.index_at(0), edge.index_at(1));
      let dist_sq = self.lengths_sq[iedge];
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
  /// The normalized Cayley-Menger determinant: equal to
  /// $det g \/ ("dim"!)^2$ as a polynomial identity in the squared lengths,
  /// on any signature. Its sign is $(-1)^q$, the parity of the signature;
  /// positive is the Euclidean(-realizable) case.
  pub fn cayley_menger_det(&self) -> f64 {
    cayley_menger_factor(self.dim()) * self.cayley_menger_matrix().determinant()
  }
  /// Whether the squared lengths are realizable by a Euclidean point
  /// configuration: the Riemannian ($q = 0$) corner of the signature range.
  pub fn is_coordinate_realizable(&self) -> bool {
    self.cayley_menger_det() >= 0.0
  }
  /// The volume $vol(hat(K)) sqrt(abs(det g)) = sqrt(abs("CM det"))$, on any
  /// signature.
  pub fn vol(&self) -> f64 {
    self.cayley_menger_det().abs().sqrt()
  }
  pub fn is_degenerate(&self) -> bool {
    self.vol() <= 1e-12
  }
}
pub fn cayley_menger_factor(dim: Dim) -> f64 {
  (-1.0f64).powi(dim as i32 + 1) / factorial(dim).pow(2) as f64 / 2f64.powi(dim as i32)
}

impl SimplexLengthsSq {
  /// Regge calculus: the squared lengths a metric tensor induces on the
  /// edges of the reference simplex, on any signature.
  ///
  /// The spanning (basis) vector $e_i$ points from vertex $0$ to vertex
  /// $i + 1$, so edges from the origin are signed squared basis norms and
  /// $norm(v_j - v_i)^2_g = g_(i-1,i-1) + g_(j-1,j-1) - 2 g_(i-1,j-1)$
  /// otherwise. Inverse of [`Self::to_metric_tensor`], with no square root
  /// anywhere: the causal sign of every edge survives.
  pub fn from_metric_tensor(metric: &Gramian) -> Self {
    let dim = metric.dim();

    let mut lengths_sq = Vector::zeros(nedges(dim));
    for (iedge, edge) in combinations(dim + 1, 2).enumerate() {
      let (vi, vj) = (edge.index_at(0), edge.index_at(1));
      lengths_sq[iedge] = if vi == 0 {
        metric.basis_inner(vj - 1, vj - 1)
      } else {
        metric.basis_inner(vi - 1, vi - 1) + metric.basis_inner(vj - 1, vj - 1)
          - 2.0 * metric.basis_inner(vi - 1, vj - 1)
      };
    }

    Self::new(lengths_sq, dim)
  }

  /// The full metric: the Gramian on tangent vectors together with its
  /// inverse on covectors, of whatever signature the squared lengths
  /// describe.
  pub fn metric(&self) -> Metric {
    Metric::new(self.to_metric_tensor())
  }

  /// Regge calculus: the metric tensor is the polarization identity in the
  /// signed squared lengths, $g_(i i) = s_(0, i+1)$ and
  /// $g_(i j) = (s_(0, i+1) + s_(0, j+1) - s_(i+1, j+1)) \/ 2$ -- rational in
  /// the Regge data and valid on any signature, which is why the squared
  /// length, not the length, is the primitive.
  pub fn to_metric_tensor(&self) -> Gramian {
    let mut metric = Matrix::zeros(self.dim(), self.dim());
    for i in 0..self.dim() {
      metric[(i, i)] = self[edge_index(0, i + 1)];
    }
    for i in 0..self.dim() {
      for j in (i + 1)..self.dim() {
        let s0i = self[edge_index(0, i + 1)];
        let s0j = self[edge_index(0, j + 1)];
        let sij = self[edge_index(i + 1, j + 1)];

        let val = 0.5 * (s0i + s0j - sij);

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

  /// from_metric_tensor and to_metric_tensor are inverse -- on every
  /// signature, the flat models pulled back to non-diagonal form included.
  /// The Regge representation loses nothing of a pseudo-Riemannian metric.
  #[test]
  fn metric_tensor_roundtrip() {
    for dim in 1..=4 {
      let lengths_sq = SimplexLengthsSq::standard(dim);
      let roundtrip = SimplexLengthsSq::from_metric_tensor(&lengths_sq.to_metric_tensor());
      assert_relative_eq!(lengths_sq.vector(), roundtrip.vector(), epsilon = 1e-12);

      for q in 0..=dim {
        let j = Matrix::from_fn(dim, dim, |i, jj| {
          if i == jj {
            1.0
          } else if i > jj {
            ((2 * i + 3 * jj) % 4) as f64 / 8.0
          } else {
            0.0
          }
        });
        let g = Gramian::pseudo_euclidean(dim - q, q).pullback(&j);
        let regge = SimplexLengthsSq::from_metric_tensor(&g);
        assert_relative_eq!(
          regge.to_metric_tensor().matrix(),
          g.matrix(),
          epsilon = 1e-12
        );
        assert_eq!(regge.to_metric_tensor().signature(), (dim - q, q));
      }
    }
  }

  /// The causal trichotomy of Regge edges on a Minkowski cell: the reference
  /// simplex measured with $eta$ has its time edge timelike, its space edges
  /// spacelike -- and the volume is the reference volume, $|det eta| = 1$.
  #[test]
  fn minkowski_regge_edges() {
    for dim in 2..=4 {
      let regge = SimplexLengthsSq::from_metric_tensor(&Gramian::minkowski(dim));
      // Edge 0-1 is the time axis $e_0$.
      assert_eq!(regge.causal_type(edge_index(0, 1)), CausalType::Timelike);
      // Edge 0-2 is the space axis $e_1$.
      assert_eq!(regge.causal_type(edge_index(0, 2)), CausalType::Spacelike);
      // Edge 1-2 is $e_1 - e_0$ with $norm^2_eta = 1 - 1 = 0$: lightlike.
      assert_eq!(regge.causal_type(edge_index(1, 2)), CausalType::Null);

      assert!(!regge.is_coordinate_realizable());
      assert_relative_eq!(
        regge.vol(),
        SimplexLengthsSq::standard(dim).vol(),
        epsilon = 1e-12
      );
    }
  }
}
