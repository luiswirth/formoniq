#![doc = include_str!("../README.md")]

extern crate nalgebra as na;

/// The dimension of a space or object.
pub type Dim = usize;

/// The signature $(p, q)$ of a non-degenerate symmetric bilinear form: the
/// number of positive and negative eigenvalues. By Sylvester's law of inertia
/// it is a basis invariant. $q = 0$ is Riemannian (positive definite);
/// $q = 1$ or $p = 1$ is Lorentzian.
pub type Signature = (usize, usize);

type Matrix = na::DMatrix<f64>;
type Vector = na::DVector<f64>;

fn is_full_rank(matrix: &Matrix, eps: f64) -> bool {
  if matrix.is_empty() {
    return true;
  }
  matrix.rank(eps) == matrix.nrows().min(matrix.ncols())
}
fn is_symmetric(matrix: &Matrix) -> bool {
  if matrix.is_empty() {
    return matrix.is_square();
  }
  (matrix - matrix.transpose()).amax() <= 1e-9 * matrix.amax().max(1.0)
}

/// The causal character of a vector under an indefinite metric, classified
/// from the sign of $g(v, v)$ in the mostly-plus convention
/// $(-, +, dots.c, +)$: negative is timelike, zero null, positive spacelike.
///
/// On a Riemannian metric every nonzero vector is spacelike; the trichotomy
/// only becomes non-trivial on an indefinite signature. The signed squared
/// norm is the primitive -- a norm $sqrt(g(v, v))$ alone cannot carry the
/// causal character.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CausalType {
  Timelike,
  Null,
  Spacelike,
}
impl CausalType {
  /// Classify the sign of a signed squared norm $g(v, v)$.
  pub fn from_norm_sq(norm_sq: f64) -> Self {
    match norm_sq
      .partial_cmp(&0.0)
      .expect("Squared norm must not be NaN.")
    {
      std::cmp::Ordering::Less => Self::Timelike,
      std::cmp::Ordering::Equal => Self::Null,
      std::cmp::Ordering::Greater => Self::Spacelike,
    }
  }
}

/// A Gram matrix: a non-degenerate symmetric bilinear form expressed in a
/// basis, of arbitrary signature $(p, q)$.
///
/// Riemannian ($q = 0$, positive definite) and Lorentzian inner products are
/// one signature-parameterized type, not two code paths: every operation is
/// defined for any signature, and the few that are signature-sensitive (the
/// volume factor, the causal trichotomy) read the signature off the form
/// itself rather than assuming it.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Gramian {
  /// Symmetric non-degenerate matrix.
  matrix: Matrix,
}
impl Gramian {
  pub fn new(matrix: Matrix) -> Self {
    assert!(is_symmetric(&matrix), "Matrix must be symmetric.");
    let this = Self { matrix };
    assert!(
      this.is_nondegenerate(),
      "Matrix must be non-degenerate (invertible)."
    );
    this
  }
  pub fn new_unchecked(matrix: Matrix) -> Self {
    if cfg!(debug_assertions) {
      Self::new(matrix)
    } else {
      Self { matrix }
    }
  }
  pub fn from_euclidean_vectors(vectors: Matrix) -> Self {
    assert!(is_full_rank(&vectors, 1e-9), "Matrix must be full rank.");
    let matrix = vectors.transpose() * vectors;
    Self::new_unchecked(matrix)
  }
  /// Orthonormal euclidean metric.
  pub fn standard(dim: Dim) -> Self {
    let matrix = Matrix::identity(dim, dim);
    Self::new_unchecked(matrix)
  }
  /// The flat pseudo-Euclidean form of signature $(p, q)$:
  /// $"diag"(+1, dots.c, +1, -1, dots.c, -1)$ with $p$ pluses followed by $q$
  /// minuses. `standard` is the case $q = 0$.
  pub fn pseudo_euclidean(p: usize, q: usize) -> Self {
    let dim = p + q;
    let mut matrix = Matrix::identity(dim, dim);
    for i in p..dim {
      matrix[(i, i)] = -1.0;
    }
    Self::new_unchecked(matrix)
  }
  /// The Minkowski metric $eta = "diag"(-1, +1, dots.c, +1)$ in the mostly-plus
  /// convention: the timelike direction is basis vector $0$, the remaining
  /// $n - 1$ are spacelike, signature $(n - 1, 1)$. The flat model of a
  /// Lorentzian manifold; its spatial block is exactly `standard(n - 1)`, which
  /// is how the Riemannian world sits inside the Lorentzian one.
  pub fn minkowski(dim: Dim) -> Self {
    assert!(dim >= 1, "Minkowski space has at least the time axis.");
    let mut matrix = Matrix::identity(dim, dim);
    matrix[(0, 0)] = -1.0;
    Self::new_unchecked(matrix)
  }

  pub fn matrix(&self) -> &Matrix {
    &self.matrix
  }
  pub fn dim(&self) -> Dim {
    self.matrix.nrows()
  }
  pub fn det(&self) -> f64 {
    self.matrix.determinant()
  }
  /// The volume factor $sqrt(abs(det g))$.
  ///
  /// For an indefinite form the determinant is negative whenever $q$ is odd,
  /// so the absolute value is part of the definition, not a safeguard: this is
  /// the density of the (pseudo-)volume form on any signature. The sign of the
  /// determinant, $(-1)^q$, is carried separately by [`Self::signature`].
  pub fn det_sqrt(&self) -> f64 {
    self.det().abs().sqrt()
  }
  /// The signature $(p, q)$: the number of positive and negative eigenvalues.
  /// The empty $0 times 0$ form has signature $(0, 0)$.
  pub fn signature(&self) -> Signature {
    if self.matrix.is_empty() {
      return (0, 0);
    }
    let eigenvalues = self.matrix.symmetric_eigenvalues();
    let p = eigenvalues.iter().filter(|&&lambda| lambda > 0.0).count();
    (p, self.dim() - p)
  }
  /// Whether the form is positive definite: the Riemannian special case
  /// $q = 0$ of the pseudo-Riemannian generality.
  pub fn is_riemannian(&self) -> bool {
    na::Cholesky::new(self.matrix.clone()).is_some()
  }
  fn is_nondegenerate(&self) -> bool {
    if self.matrix.is_empty() {
      return true;
    }
    let eigenvalues = self.matrix.symmetric_eigenvalues();
    let scale = eigenvalues.amax();
    scale > 0.0
      && eigenvalues
        .iter()
        .all(|lambda| lambda.abs() > 1e-12 * scale)
  }
  pub fn inverse(self) -> Self {
    let matrix = self
      .matrix
      .try_inverse()
      .expect("Non-degenerate is always invertible.");
    Self::new_unchecked(matrix)
  }

  /// The pullback $J^top G J$ of the metric along a linear map $J$.
  ///
  /// $G$ is a covariant 2-tensor, and this is its pullback: if $J: U -> V$
  /// sends a basis of $U$ to vectors of $V$, the result is the Gramian $U$
  /// inherits by measuring those images with $G$. For definite $G$, injective
  /// $J$ (full column rank) keeps the result a metric; for indefinite $G$ the
  /// pullback onto a proper subspace can be degenerate (a null subspace), so
  /// only square invertible $J$ -- e.g. the affine child-cell Jacobians of a
  /// simplex subdivision -- is guaranteed to stay a metric, with the same
  /// signature by Sylvester's law of inertia.
  pub fn pullback(&self, jacobian: &Matrix) -> Self {
    Self::new_unchecked(self.inner_mat(jacobian, jacobian))
  }
}

/// Inner product functionality expressed directly in terms of the basis.
impl Gramian {
  pub fn basis_inner(&self, i: usize, j: usize) -> f64 {
    self.matrix[(i, j)]
  }
  pub fn basis_norm_sq(&self, i: usize) -> f64 {
    self.basis_inner(i, i)
  }
  pub fn basis_norm(&self, i: usize) -> f64 {
    self.basis_norm_sq(i).abs().sqrt()
  }
  /// Cosine of the angle between basis vectors `i` and `j`.
  ///
  /// An angle presupposes a definite metric; on an indefinite one the
  /// Cauchy-Schwarz bound fails and this quotient is not a cosine.
  pub fn basis_angle_cos(&self, i: usize, j: usize) -> f64 {
    self.basis_inner(i, j) / self.basis_norm(i) / self.basis_norm(j)
  }
  /// Angle (in radians) between basis vectors `i` and `j`.
  /// Meaningful on a definite metric only, like [`Self::basis_angle_cos`].
  pub fn basis_angle(&self, i: usize, j: usize) -> f64 {
    self.basis_angle_cos(i, j).acos()
  }

  /// Squared distance between two of the $"dim"+1$ points this Gramian is the
  /// edge metric of: vertex $0$ (the Gramian's own origin) and its $"dim"$
  /// basis vectors, read as vertices $1..="dim"$. $d(0,j)^2$ is the basis
  /// vector's own norm; $d(i,j)^2$ for $i,j >= 1$ follows from the law of
  /// cosines applied to the two edges from the origin,
  /// $d(i,j)^2 = g_(i-1,i-1) + g_(j-1,j-1) - 2 g_(i-1,j-1)$.
  fn vertex_dist_sq(&self, i: usize, j: usize) -> f64 {
    if i == j {
      0.0
    } else if i == 0 {
      self.basis_norm_sq(j - 1)
    } else if j == 0 {
      self.basis_norm_sq(i - 1)
    } else {
      self.basis_norm_sq(i - 1) + self.basis_norm_sq(j - 1) - 2.0 * self.basis_inner(i - 1, j - 1)
    }
  }

  /// The interior angle at vertex `v`, between its edges to vertices `a` and
  /// `b` (all in $0..="dim"$, vertex $0$ the Gramian's own origin): the law
  /// of cosines applied to the squared distance between any two of the
  /// simplex's vertices (vertex $0$ the Gramian's own origin). Generalizes
  /// [`Self::basis_angle`] (the case $v = 0$) to any of the simplex's
  /// $"dim"+1$ corners, not just the one the Gramian is based at.
  pub fn vertex_angle(&self, v: usize, a: usize, b: usize) -> f64 {
    let d_va = self.vertex_dist_sq(v, a);
    let d_vb = self.vertex_dist_sq(v, b);
    let d_ab = self.vertex_dist_sq(a, b);
    ((d_va + d_vb - d_ab) / (2.0 * (d_va * d_vb).sqrt())).acos()
  }
}
impl std::ops::Index<(usize, usize)> for Gramian {
  type Output = f64;
  fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
    &self.matrix[(i, j)]
  }
}

/// A pseudo-Riemannian metric: the Gramian on tangent vectors together with
/// its inverse, the induced Gramian on covectors.
///
/// One signature-parameterized type for every non-degenerate symmetric metric:
/// Riemannian geometry is the special case $q = 0$, Lorentzian geometry the
/// case $q = 1$ (mostly-plus), with no separate code path for either.
///
/// Keeping both Gramians eliminates the recurring question of whether a
/// computation needs $g$ or $g^(-1)$: contravariant quantities (vectors) are
/// measured by [`Self::vector_gramian`], covariant ones (forms) by
/// [`Self::covector_gramian`] -- indefinite or not.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PseudoRiemannianMetric {
  vector_gramian: Gramian,
  covector_gramian: Gramian,
}
impl PseudoRiemannianMetric {
  pub fn new(vector_gramian: Gramian) -> Self {
    let covector_gramian = vector_gramian.clone().inverse();
    Self {
      vector_gramian,
      covector_gramian,
    }
  }
  /// Orthonormal euclidean metric.
  pub fn standard(dim: Dim) -> Self {
    Self {
      vector_gramian: Gramian::standard(dim),
      covector_gramian: Gramian::standard(dim),
    }
  }
  /// The Minkowski metric $eta$ (mostly-plus, time along basis vector $0$):
  /// the flat Lorentzian metric. $eta^(-1) = eta$, so both Gramians are the
  /// same matrix.
  pub fn minkowski(dim: Dim) -> Self {
    Self {
      vector_gramian: Gramian::minkowski(dim),
      covector_gramian: Gramian::minkowski(dim),
    }
  }

  pub fn dim(&self) -> Dim {
    self.vector_gramian.dim()
  }
  /// The metric tensor $g$: the inner product on tangent vectors.
  pub fn vector_gramian(&self) -> &Gramian {
    &self.vector_gramian
  }
  /// The inverse metric tensor $g^(-1)$: the inner product on covectors.
  pub fn covector_gramian(&self) -> &Gramian {
    &self.covector_gramian
  }
  /// The volume factor $sqrt(abs(det g))$ of the metric: the density of the
  /// (pseudo-)volume form, on any signature.
  pub fn det_sqrt(&self) -> f64 {
    self.vector_gramian.det_sqrt()
  }
  /// The signature $(p, q)$ of the metric. The covector Gramian shares it:
  /// inversion preserves eigenvalue signs.
  pub fn signature(&self) -> Signature {
    self.vector_gramian.signature()
  }
  /// Whether the metric is Riemannian: the positive-definite special case
  /// $q = 0$.
  pub fn is_riemannian(&self) -> bool {
    self.vector_gramian.is_riemannian()
  }

  /// The pullback of the metric along a linear map $J$ of tangent spaces:
  /// the metric $J^top g J$ that a domain inherits by pushing its vectors
  /// through $J$ and measuring them with $g$. The covector Gramian is the
  /// inverse, recomputed rather than pushed forward. For an affine subcell of
  /// a flat cell, $J$ is the cell's constant Jacobian and this is the subcell's
  /// exact metric, of the same signature (Sylvester); for indefinite $g$ a
  /// non-square $J$ can land on a degenerate (null) subspace, which is no
  /// longer a metric -- see [`Gramian::pullback`].
  pub fn pullback(&self, jacobian: &Matrix) -> Self {
    Self::new(self.vector_gramian.pullback(jacobian))
  }
}

/// Metric operations on tangent vectors (contravariant quantities), measured
/// by the metric tensor $g$. These are the canonical measurements of
/// directions and magnitudes; the dual operations on covectors are available
/// through [`Self::covector_gramian`].
impl PseudoRiemannianMetric {
  /// Inner product $g(v, w)$ of two tangent vectors.
  pub fn inner(&self, v: &Vector, w: &Vector) -> f64 {
    self.vector_gramian.inner(v, w)
  }
  /// Signed squared length $g(v, v)$ of a tangent vector: the primitive that
  /// stays well defined on any signature, and whose sign is the causal
  /// character.
  pub fn norm_sq(&self, v: &Vector) -> f64 {
    self.vector_gramian.norm_sq(v)
  }
  /// Magnitude $sqrt(abs(g(v, v)))$ of a tangent vector. On an indefinite
  /// metric this is meaningful only together with the causal character
  /// ([`Self::causal_type`]); it is never NaN.
  pub fn norm(&self, v: &Vector) -> f64 {
    self.vector_gramian.norm(v)
  }
  /// The causal character of a tangent vector: the sign of $g(v, v)$,
  /// in the mostly-plus convention of [`CausalType`].
  pub fn causal_type(&self, v: &Vector) -> CausalType {
    self.vector_gramian.causal_type(v)
  }
  /// Cosine of the angle between two tangent vectors.
  /// Meaningful on a Riemannian (definite) metric only.
  pub fn angle_cos(&self, v: &Vector, w: &Vector) -> f64 {
    self.vector_gramian.angle_cos(v, w)
  }
  /// Angle (in radians) between two tangent vectors.
  /// Meaningful on a Riemannian (definite) metric only.
  pub fn angle(&self, v: &Vector, w: &Vector) -> f64 {
    self.vector_gramian.angle(v, w)
  }
}

/// Inner product functionality on arbitrary elements.
impl Gramian {
  pub fn inner(&self, v: &Vector, w: &Vector) -> f64 {
    (v.transpose() * self.matrix() * w).x
  }
  /// Gram matrix of two families of vectors (given as columns): `vᵀ G w`.
  pub fn inner_mat(&self, v: &Matrix, w: &Matrix) -> Matrix {
    v.transpose() * self.matrix() * w
  }
  /// Signed squared norm $g(v, v)$: the primitive, well defined and signed on
  /// any signature.
  pub fn norm_sq(&self, v: &Vector) -> f64 {
    self.inner(v, v)
  }
  /// Elementwise signed squared norms of a family of vectors (given as columns).
  pub fn norm_sq_mat(&self, v: &Matrix) -> Matrix {
    self.inner_mat(v, v)
  }
  /// Magnitude $sqrt(abs(g(v, v)))$. On an indefinite metric this is
  /// meaningful only together with the causal character
  /// ([`Self::causal_type`]); it is never NaN.
  pub fn norm(&self, v: &Vector) -> f64 {
    self.norm_sq(v).abs().sqrt()
  }
  /// Elementwise magnitudes of a family of vectors (given as columns).
  pub fn norm_mat(&self, v: &Matrix) -> Matrix {
    self.inner_mat(v, v).map(|x| x.abs().sqrt())
  }
  /// The causal character of `v`: the sign of $g(v, v)$, in the mostly-plus
  /// convention of [`CausalType`].
  pub fn causal_type(&self, v: &Vector) -> CausalType {
    CausalType::from_norm_sq(self.norm_sq(v))
  }
  /// Cosine of the angle between `v` and `w`.
  /// Meaningful on a definite metric only.
  pub fn angle_cos(&self, v: &Vector, w: &Vector) -> f64 {
    self.inner(v, w) / self.norm(v) / self.norm(w)
  }
  /// Angle (in radians) between `v` and `w`.
  /// Meaningful on a definite metric only.
  pub fn angle(&self, v: &Vector, w: &Vector) -> f64 {
    self.angle_cos(v, w).acos()
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::f64::consts::FRAC_PI_2;

  #[test]
  fn euclidean_angles_and_norms() {
    let g = Gramian::standard(2);
    let e0 = Vector::from_column_slice(&[1.0, 0.0]);
    let e1 = Vector::from_column_slice(&[0.0, 1.0]);

    assert!((g.norm(&e0) - 1.0).abs() < 1e-12);
    assert!((g.angle(&e0, &e1) - FRAC_PI_2).abs() < 1e-12);
    assert!(g.angle_cos(&e0, &e1).abs() < 1e-12);
    assert!(g.angle(&e0, &e0).abs() < 1e-12);
  }

  #[test]
  fn nonstandard_metric_angle_matches_definition() {
    // A metric that stretches the second axis; the coordinate axes stay
    // g-orthogonal, but a diagonal vector no longer bisects them.
    let g = Gramian::new(Matrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 4.0]));
    let v = Vector::from_column_slice(&[1.0, 1.0]);
    let w = Vector::from_column_slice(&[1.0, 0.0]);

    // g(v, w) = 1, |v|_g = sqrt(5), |w|_g = 1.
    assert!((g.inner(&v, &w) - 1.0).abs() < 1e-12);
    assert!((g.norm(&v) - 5.0_f64.sqrt()).abs() < 1e-12);
    assert!((g.angle_cos(&v, &w) - 1.0 / 5.0_f64.sqrt()).abs() < 1e-12);
  }

  // 2 on the diagonal, 1 off it: SPD with eigenvalues `dim+1` (once) and 1.
  fn spd(dim: Dim) -> Gramian {
    let mut m = Matrix::from_element(dim, dim, 1.0);
    for i in 0..dim {
      m[(i, i)] = 2.0;
    }
    Gramian::new_unchecked(m)
  }

  // A deterministic full-column-rank `nrows x ncols` matrix (ncols <= nrows):
  // unit lower-triangular columns, injective, so the pullback stays s.p.d.
  fn full_col_rank(nrows: usize, ncols: usize) -> Matrix {
    Matrix::from_fn(nrows, ncols, |i, j| {
      if i == j {
        1.0
      } else if i > j {
        0.5
      } else {
        0.0
      }
    })
  }

  fn close(a: &Matrix, b: &Matrix) {
    assert_eq!(a.shape(), b.shape());
    assert!((a - b).amax() < 1e-9, "{a} != {b}");
  }

  /// The pullback is functorial: pulling $g$ back along a composite $A B$
  /// equals pulling first along $A$, then along $B$,
  /// $(A B)^* g = B^* (A^* g)$, i.e. $(A B)^top g (A B) = B^top (A^top g A) B$.
  /// Swept over the full grade/dimension range including the degenerate
  /// zero-column maps, where the pulled-back metric is the empty $0 times 0$
  /// Gramian.
  #[test]
  fn pullback_is_functorial() {
    for n in 0..=4 {
      let g = spd(n);
      for k in 0..=n {
        let a = full_col_rank(n, k);
        for m in 0..=k {
          let b = full_col_rank(k, m);
          let composite = &a * &b;
          let lhs = g.pullback(&composite);
          let rhs = g.pullback(&a).pullback(&b);
          close(lhs.matrix(), rhs.matrix());
        }
      }
    }
  }

  /// The pullback is literally $J^top G J$, and pulling back along the identity
  /// changes nothing.
  #[test]
  fn pullback_matches_definition_and_fixes_identity() {
    for n in 1..=4 {
      let g = spd(n);
      let j = full_col_rank(n, n);
      close(g.pullback(&j).matrix(), &(j.transpose() * g.matrix() * &j));
      close(g.pullback(&Matrix::identity(n, n)).matrix(), g.matrix());
    }
  }

  /// A pulled-back pseudo-Riemannian metric recomputes its covector Gramian as
  /// the inverse of the pulled-back vector Gramian, rather than pushing the old
  /// inverse forward.
  #[test]
  fn metric_pullback_inverts_the_pulled_back_metric() {
    for n in 1..=4 {
      let metric = PseudoRiemannianMetric::new(spd(n));
      let j = full_col_rank(n, n);
      let pulled = metric.pullback(&j);
      let expected_covector = pulled.vector_gramian().clone().inverse();
      close(
        pulled.covector_gramian().matrix(),
        expected_covector.matrix(),
      );
    }
  }

  #[test]
  fn metric_measures_tangent_vectors_with_g() {
    let g = Gramian::new(Matrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 3.0]));
    let metric = PseudoRiemannianMetric::new(g.clone());
    let v = Vector::from_column_slice(&[1.0, 1.0]);
    let w = Vector::from_column_slice(&[1.0, -1.0]);

    // Convenience methods on the metric agree with the tangent (vector) Gramian.
    assert!((metric.inner(&v, &w) - g.inner(&v, &w)).abs() < 1e-12);
    assert!((metric.norm(&v) - g.norm(&v)).abs() < 1e-12);
    assert!((metric.angle(&v, &w) - g.angle(&v, &w)).abs() < 1e-12);
  }

  /// The flat models carry their signature by construction: signature
  /// $(p, q)$, determinant sign $(-1)^q$, unit volume factor -- swept over
  /// every signature up to dimension 4, the Euclidean $q = 0$ and the empty
  /// $0 times 0$ form included.
  #[test]
  fn flat_model_signatures() {
    for dim in 0..=4 {
      for q in 0..=dim {
        let p = dim - q;
        let g = Gramian::pseudo_euclidean(p, q);
        assert_eq!(g.signature(), (p, q));
        assert_eq!(g.is_riemannian(), q == 0);
        if dim > 0 {
          assert_eq!(g.det().signum(), (-1.0f64).powi(q as i32));
        }
        assert!((g.det_sqrt() - 1.0).abs() < 1e-12);
      }
    }
  }

  /// Sylvester's law of inertia: the signature is invariant under congruence,
  /// i.e. under pullback along any invertible map.
  #[test]
  fn signature_is_congruence_invariant() {
    for dim in 1..=4 {
      for q in 0..=dim {
        let g = Gramian::pseudo_euclidean(dim - q, q);
        let j = full_col_rank(dim, dim);
        assert_eq!(g.pullback(&j).signature(), (dim - q, q));
      }
    }
  }

  /// The causal trichotomy on Minkowski space: the time axis is timelike, the
  /// space axes spacelike, the light-cone diagonal null -- and the magnitude
  /// is never NaN, on either side of the cone.
  #[test]
  fn minkowski_causal_types() {
    let eta = Gramian::minkowski(4);
    assert_eq!(eta.signature(), (3, 1));

    let e0 = Vector::from_column_slice(&[1.0, 0.0, 0.0, 0.0]);
    let e1 = Vector::from_column_slice(&[0.0, 1.0, 0.0, 0.0]);
    let light = Vector::from_column_slice(&[1.0, 1.0, 0.0, 0.0]);

    assert_eq!(eta.causal_type(&e0), CausalType::Timelike);
    assert_eq!(eta.causal_type(&e1), CausalType::Spacelike);
    assert_eq!(eta.causal_type(&light), CausalType::Null);

    assert!((eta.norm_sq(&e0) - -1.0).abs() < 1e-12);
    assert!((eta.norm(&e0) - 1.0).abs() < 1e-12);
    assert!((eta.norm(&light) - 0.0).abs() < 1e-12);

    let metric = PseudoRiemannianMetric::minkowski(4);
    assert_eq!(metric.signature(), (3, 1));
    assert_eq!(metric.causal_type(&e0), CausalType::Timelike);
    close(
      metric.covector_gramian().matrix(),
      metric.vector_gramian().matrix(),
    );
  }

  /// A symmetric but degenerate matrix is not a metric.
  #[test]
  #[should_panic(expected = "non-degenerate")]
  fn degenerate_is_rejected() {
    Gramian::new(Matrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 0.0]));
  }

  /// A non-symmetric matrix is not a bilinear form of ours.
  #[test]
  #[should_panic(expected = "symmetric")]
  fn nonsymmetric_is_rejected() {
    Gramian::new(Matrix::from_row_slice(2, 2, &[1.0, 1.0, 0.0, 1.0]));
  }
}
