use super::{EdgeIdx, LengthsSq};
use simplicial::{
  Dim,
  topology::simplex::{edge_index, nedges},
};

use metric::Metric;
use multialgebra::tensor::Slots;
use multialgebra::{Factor, Slot, Tensor, Variance};
use multiindex::{combinations, factorial};
use simplicial::linalg::{Matrix, Vector};

/// The signed squared edge lengths of a simplex: Regge calculus, on any
/// metric signature.
///
/// The squared length $s_(i j) = norm(v_j - v_i)^2_g$ is the Regge primitive
///, signed, exactly like [`Metric::norm_sq`]: positive on a spacelike
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
  /// [`Simplex::subsimps`](simplicial::topology::simplex::Simplex::subsimps) with dim 1.
  lengths_sq: Vector,
  /// Dimension of the simplex.
  dim: Dim,
}

impl SimplexLengthsSq {
  /// The invariant is non-degeneracy of the induced metric tensor, the
  /// squared lengths must describe a simplex of some signature, Euclidean
  /// realizability ([`Self::is_coordinate_realizable`]) being the Riemannian
  /// special case, not the requirement.
  pub fn new(lengths_sq: Vector, dim: impl Into<Dim>) -> Self {
    let dim = dim.into();
    assert_eq!(lengths_sq.len(), nedges(dim), "Wrong number of edges.");
    let this = Self { lengths_sq, dim };
    assert!(
      !this.is_degenerate(),
      "Simplex metric must be non-degenerate."
    );
    this
  }
  pub fn new_unchecked(lengths_sq: Vector, dim: impl Into<Dim>) -> Self {
    let dim = dim.into();
    if cfg!(debug_assertions) {
      Self::new(lengths_sq, dim)
    } else {
      Self { lengths_sq, dim }
    }
  }
  /// The unit simplex: edges at the origin vertex are unit, all others
  /// connect two standard basis vertices with squared length $2$.
  pub fn unit(dim: impl Into<Dim>) -> SimplexLengthsSq {
    let dim = dim.into();
    let lengths_sq: Vec<f64> = combinations((dim + 1).index(), 2)
      .map(|edge| if edge.contains(0) { 1.0 } else { 2.0 })
      .collect();

    Self::new(lengths_sq.into(), dim)
  }
  pub fn dim(&self) -> Dim {
    self.dim
  }
  pub fn nvertices(&self) -> usize {
    self.dim().index() + 1
  }
  /// The diameter of this cell: the largest edge magnitude, which by
  /// convexity bounds the distance of any two points inside. A metric
  /// quantity of the Riemannian case. On an indefinite metric it is a mesh
  /// scale, not a distance.
  pub fn diameter(&self) -> f64 {
    self.max_length()
  }

  /// The shape regularity measure of this cell.
  pub fn shape_regularity(&self) -> f64 {
    self.volume_scale() / self.vol()
  }
  /// The order of magnitude a non-degenerate volume has at this diameter,
  /// $"diam"^n$: the scale against which a volume is small or large.
  pub fn volume_scale(&self) -> f64 {
    self.diameter().powi(self.dim().index() as i32)
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
}

impl LengthsSq for SimplexLengthsSq {
  fn lengths_sq(&self) -> &Vector {
    &self.lengths_sq
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
  /// The signed squared distance between two vertices: the stored Regge datum,
  /// and zero along the diagonal.
  pub fn vertex_dist_sq(&self, vi: usize, vj: usize) -> f64 {
    if vi == vj {
      0.0
    } else {
      self[edge_index(vi, vj)]
    }
  }

  /// The interior angle at vertex `v` between its edges to `a` and `b`, by the
  /// law of cosines on that corner's three squared lengths.
  ///
  /// Intrinsic, needing neither coordinates nor a Gramian: an angle is a
  /// function of three edge lengths, which is the Regge datum itself.
  ///
  /// An angle presupposes a definite corner, so on a Lorentzian simplex the
  /// quotient leaves $[-1, 1]$ and the result is NaN, the same failure of the
  /// Cauchy-Schwarz bound that makes an angle meaningless on an indefinite
  /// form.
  pub fn vertex_angle(&self, v: usize, a: usize, b: usize) -> f64 {
    let (d_va, d_vb, d_ab) = (
      self.vertex_dist_sq(v, a),
      self.vertex_dist_sq(v, b),
      self.vertex_dist_sq(a, b),
    );
    ((d_va + d_vb - d_ab) / (2.0 * (d_va * d_vb).sqrt())).acos()
  }

  /// The matrix of signed squared distances between the vertices: exactly the
  /// stored Regge data, symmetrized.
  pub fn distance_matrix(&self) -> Matrix {
    Matrix::from_fn(self.nvertices(), self.nvertices(), |vi, vj| {
      self.vertex_dist_sq(vi, vj)
    })
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
  /// Whether the induced metric is degenerate: the volume vanishes against
  /// [`Self::volume_scale`], the volume a simplex of this diameter would
  /// otherwise have.
  ///
  /// The comparison is relative, so the predicate is invariant under a uniform
  /// scaling of the geometry, as degeneracy itself is. An absolute bound on the
  /// volume would instead call every simplex of a fine mesh degenerate, the
  /// more so the higher the dimension.
  pub fn is_degenerate(&self) -> bool {
    self.vol() <= DEGENERACY_FLOOR * self.volume_scale()
  }
}

/// The relative floor a volume must clear to count as non-degenerate: the
/// reciprocal of the shape regularity a simplex may reach before its metric is
/// numerically singular.
const DEGENERACY_FLOOR: f64 = 1e-12;
pub fn cayley_menger_factor(dim: impl Into<Dim>) -> f64 {
  let dim = dim.into();
  (-1.0f64).powi(dim.index() as i32 + 1)
    / factorial(dim.index()).pow(2) as f64
    / 2f64.powi(dim.index() as i32)
}

/// The symmetric square $u_e dot.circle u_e$ of each edge vector of an
/// $n$-simplex, in edge order: a basis of $"Sym"^2$ indexed by the edges.
///
/// The dimension count is not a coincidence. An
/// $n$-simplex has $binom(n+1, 2) = n(n+1)\/2$ edges and
/// $dim "Sym"^2(RR^n) = n(n+1)\/2$, the same number, and these squares are
/// linearly independent, hence a basis. Squared edge lengths are therefore
/// exactly the components of the metric in the basis dual to this one:
/// $s_e = angle.l g, u_e dot.circle u_e angle.r$, and
/// [`SimplexLengthsSq::metric`] is that change of basis, the polarization
/// identity being what it looks like written out.
///
/// It is also why the geometry of a subsimplex costs nothing (invariant 2). A
/// face's edges are a subset of the simplex's, so restricting the metric to a
/// face is selecting the components indexed by that face's edges, an index
/// selection, where the cartesian frame would need a projection. Totality over
/// every grade is a property of this basis rather than of the code.
///
/// Metric-free: the edge vectors are read off the chart, $e_(i-1)$ pointing from
/// vertex $0$ to vertex $i$, and no length is taken. A function of `dim` alone,
/// like every other datum of the reference cell.
pub fn unit_edge_squares(dim: impl Into<Dim>) -> Vec<Tensor> {
  let dim = dim.into();
  let n = dim.index();
  combinations(n + 1, 2)
    .map(|edge| {
      let (vi, vj) = (edge.index_at(0), edge.index_at(1));
      // The edge from vi to vj in the chart's frame, vertex 0 being the origin.
      let mut u = Vector::zeros(n);
      if vi > 0 {
        u[vi - 1] -= 1.0;
      }
      if vj > 0 {
        u[vj - 1] += 1.0;
      }
      let line = |u: Vector| {
        Tensor::new(
          Slots::from_iter([Slot::new(Factor::symmetric(1), Variance::Contravariant, n)]),
          u,
        )
      };
      line(u.clone()).product(&line(u))
    })
    .collect()
}

impl SimplexLengthsSq {
  /// Regge calculus: the squared lengths a metric tensor induces on the
  /// edges of the unit simplex, on any signature.
  ///
  /// The spanning (basis) vector $e_i$ points from vertex $0$ to vertex
  /// $i + 1$, so edges from the origin are signed squared basis norms and
  /// $norm(v_j - v_i)^2_g = g_(i-1,i-1) + g_(j-1,j-1) - 2 g_(i-1,j-1)$
  /// otherwise. Inverse of [`Self::metric`], with no square root
  /// anywhere: the causal sign of every edge survives.
  pub fn from_metric(metric: &Metric) -> Self {
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

  /// Regge calculus: the metric tensor is the polarization identity in the
  /// signed squared lengths, $g_(i i) = s_(0, i+1)$ and
  /// $g_(i j) = (s_(0, i+1) + s_(0, j+1) - s_(i+1, j+1)) \/ 2$, rational in
  /// the Regge data and valid on any signature, which is why the squared
  /// length, not the length, is the primitive.
  ///
  /// Both hypotheses of [`Metric`] hold by construction here rather than by
  /// inspection, so this builds unchecked. Symmetry is structural: the
  /// polarization writes each off-diagonal pair from a single value. And
  /// non-degeneracy is this type's own constructor invariant, already
  /// established when the lengths were built, so re-deriving it would run a
  /// symmetric eigendecomposition once per cell to reconfirm a proof already
  /// in hand. The debug build still asserts both.
  pub fn metric(&self) -> Metric {
    let mut metric = Matrix::zeros(self.dim().index(), self.dim().index());
    for i in 0..self.dim().index() {
      metric[(i, i)] = self[edge_index(0, i + 1)];
    }
    for i in 0..self.dim().index() {
      for j in (i + 1)..self.dim().index() {
        let s0i = self[edge_index(0, i + 1)];
        let s0j = self[edge_index(0, j + 1)];
        let sij = self[edge_index(i + 1, j + 1)];

        let val = 0.5 * (s0i + s0j - sij);

        metric[(i, j)] = val;
        metric[(j, i)] = val;
      }
    }
    Metric::new(Variance::Covariant, metric)
  }
}
#[cfg(test)]
mod test {
  use super::*;
  use metric::CausalType;
  use multialgebra::tensor::pairing;
  use multiindex::Dim;

  use approx::assert_relative_eq;

  /// A metric with distinct entries throughout: an equal-entry probe hides a
  /// wrong weight in a basis change.
  fn probe_metric(dim: usize) -> Metric {
    let a = Matrix::from_fn(dim, dim, |i, j| ((3 * i + 7 * j) % 5) as f64 / 5.0);
    Metric::new(
      Variance::Covariant,
      a.transpose() * &a + Matrix::identity(dim, dim),
    )
  }

  /// from_metric and metric are inverse, on every
  /// signature, the flat models pulled back to non-diagonal form included.
  /// The Regge representation loses nothing of a pseudo-Riemannian metric.
  #[test]
  fn metric_tensor_roundtrip() {
    for dim in (1..=4usize).map(Dim::from) {
      let lengths_sq = SimplexLengthsSq::unit(dim);
      let roundtrip = SimplexLengthsSq::from_metric(&lengths_sq.metric());
      assert_relative_eq!(lengths_sq.vector(), roundtrip.vector(), epsilon = 1e-12);

      for q in 0..=dim.index() {
        let j = Matrix::from_fn(dim.index(), dim.index(), |i, jj| {
          if i == jj {
            1.0
          } else if i > jj {
            ((2 * i + 3 * jj) % 4) as f64 / 8.0
          } else {
            0.0
          }
        });
        let g = Metric::pseudo_euclidean(dim.index() - q, q).pullback(&j);
        let regge = SimplexLengthsSq::from_metric(&g);
        assert_relative_eq!(regge.metric().matrix(), g.matrix(), epsilon = 1e-12);
        assert_eq!(regge.metric().signature(), (dim.index() - q, q));
      }
    }
  }

  /// Degeneracy is measured against the simplex's own scale, so a uniform
  /// scaling leaves it alone. Collapsing two vertices onto each other, which
  /// makes two rows of the distance matrix agree, trips it at every scale.
  #[test]
  fn degeneracy_is_scale_invariant() {
    for dim in (2..=4usize).map(Dim::from) {
      for scale in [1e-4, 1.0, 1e4] {
        let mut lengths = SimplexLengthsSq::unit(dim);
        *lengths.vector_mut() *= scale * scale;
        assert!(!lengths.is_degenerate());

        lengths.vector_mut()[edge_index(1, 2)] = 0.0;
        assert!(lengths.is_degenerate());
      }
    }
  }

  /// The causal trichotomy of Regge edges on a Minkowski cell: the reference
  /// simplex measured with $eta$ has its time edge timelike, its space edges
  /// spacelike, and the volume is the reference volume, $|det eta| = 1$.
  #[test]
  fn minkowski_regge_edges() {
    for dim in (2..=4usize).map(Dim::from) {
      let regge = SimplexLengthsSq::from_metric(&Metric::minkowski(dim.index()));
      // Edge 0-1 is the time axis $e_0$.
      assert_eq!(regge.causal_type(edge_index(0, 1)), CausalType::Timelike);
      // Edge 0-2 is the space axis $e_1$.
      assert_eq!(regge.causal_type(edge_index(0, 2)), CausalType::Spacelike);
      // Edge 1-2 is $e_1 - e_0$ with $norm^2_eta = 1 - 1 = 0$: lightlike.
      assert_eq!(regge.causal_type(edge_index(1, 2)), CausalType::Null);

      assert!(!regge.is_coordinate_realizable());
      assert_relative_eq!(
        regge.vol(),
        SimplexLengthsSq::unit(dim).vol(),
        epsilon = 1e-12
      );
    }
  }

  /// The edge squares are a basis of $"Sym"^2$: as many as its dimension,
  /// and independent. The count is $binom(n+1,2) = n(n+1)\/2 = dim "Sym"^2(RR^n)$,
  /// so a rank check is what separates "the right number" from "a basis".
  #[test]
  fn the_edge_squares_are_a_basis_of_sym2() {
    for dim in (0..=4usize).map(Dim::from) {
      let squares = unit_edge_squares(dim);
      let sym2 = Factor::symmetric(2).multidim(dim);
      assert_eq!(squares.len(), nedges(dim));
      assert_eq!(squares.len(), sym2);

      // The point simplex has no edges and a zero-dimensional Sym^2, so the
      // empty family is its basis: the count above is the whole statement and
      // there is no rank to take.
      if sym2 > 0 {
        let components = Matrix::from_fn(sym2, squares.len(), |i, e| squares[e].components()[i]);
        assert_eq!(
          components.rank(1e-9),
          sym2,
          "the squares must be independent"
        );
      }
    }
  }

  /// Squared edge lengths are the components of the metric in the basis dual to
  /// the edge squares: $s_e = angle.l g, u_e dot.circle u_e angle.r$.
  ///
  /// The polarization identity of [`SimplexLengthsSq::metric`] and
  /// [`SimplexLengthsSq::from_metric`] is that change of basis, and this is
  /// what says so rather than asserting it in prose. Swept over every signature,
  /// since the pairing is metric-free and so must hold on all of them.
  #[test]
  fn the_squared_lengths_are_the_metric_paired_with_the_edge_squares() {
    for dim in (1..=4usize).map(Dim::from) {
      for q in 0..=dim.index() {
        let metric = Metric::pseudo_euclidean(dim.index() - q, q);
        let lengths = SimplexLengthsSq::from_metric(&metric);
        for (iedge, square) in unit_edge_squares(dim).iter().enumerate() {
          assert_relative_eq!(
            lengths[iedge],
            pairing(&metric.tensor(), square),
            epsilon = 1e-12
          );
        }
      }
    }
  }

  /// Restricting the geometry to a face is an index selection in the edge
  /// basis: the face's squared lengths are the parent's at the face's own edge
  /// indices, with nothing computed.
  ///
  /// This is why geometry is defined on every simplex and not only on the cells
  /// (invariant 2). In the cartesian frame the same restriction is a projection
  /// $J^top g J$; here the two agree, which is the statement that the edge basis
  /// is the one adapted to the face lattice.
  #[test]
  fn restricting_to_a_face_selects_edge_components() {
    for dim in (1..=4usize).map(Dim::from) {
      let metric = probe_metric(dim.index());
      let lengths = SimplexLengthsSq::from_metric(&metric);

      for face in combinations(dim.index() + 1, dim.index()) {
        // The face's own squared lengths, read off the parent by index alone.
        let selected: Vec<f64> = combinations(face.card(), 2)
          .map(|pair| {
            lengths[edge_index(
              face.index_at(pair.index_at(0)),
              face.index_at(pair.index_at(1)),
            )]
          })
          .collect();
        let face_lengths = SimplexLengthsSq::new(selected.into(), dim.index() - 1);

        // The same restriction done the cartesian way: pull the metric back
        // along the inclusion of the face's spanning vectors.
        let inclusion = Matrix::from_fn(dim.index(), dim.index() - 1, |i, k| {
          let (base, other) = (face.index_at(0), face.index_at(k + 1));
          let mut column = 0.0;
          if other == i + 1 {
            column += 1.0;
          }
          if base == i + 1 {
            column -= 1.0;
          }
          column
        });
        let pulled = SimplexLengthsSq::from_metric(&metric.pullback(&inclusion));

        assert_relative_eq!(face_lengths.vector(), pulled.vector(), epsilon = 1e-12);
      }
    }
  }
}
