#![doc = include_str!("../README.md")]

extern crate nalgebra as na;

use gramian::{Gramian, Metric};
use multiindex::{binomial, combinations, Combination, Sign};

use std::marker::PhantomData;

pub use multiindex::Dim;
pub type ExteriorGrade = usize;

pub type Vector<T = f64> = na::DVector<T>;
pub type Matrix<T = f64> = na::DMatrix<T>;

/// A basis blade $e_(i_1) wedge dots.c wedge e_(i_k)$ of the exterior
/// algebra: a strictly increasing multi-index.
pub type Blade = Combination;

pub fn exterior_dim(dim: Dim, grade: ExteriorGrade) -> usize {
  binomial(dim, grade)
}

/// The basis blades of $Lambda^k (RR^n)$ in colexicographic order:
/// the order of the coefficients of an [`ExteriorElement`].
pub fn exterior_bases(dim: Dim, grade: ExteriorGrade) -> impl Iterator<Item = Blade> {
  combinations(dim, grade)
}

/// The variance of an exterior element: whether it lives in $Lambda^k V$
/// (contravariant: multivectors) or $Lambda^k V^*$ (covariant: multiforms).
///
/// A type-level marker that makes the duality pairing, the direction of the
/// functor (pushforward vs. pullback), the musical isomorphisms and the
/// choice of metric Gramian ($g$ vs. $g^(-1)$) correct by construction.
pub trait Variance: Copy + std::fmt::Debug + 'static {
  /// The dual variance: what this variance pairs against.
  type Dual: Variance<Dual = Self>;

  /// The Gramian measuring elements of this variance:
  /// the metric tensor $g$ for multivectors, its inverse $g^(-1)$ for
  /// multiforms.
  fn gramian(metric: &Metric) -> &Gramian;
}

/// The variance of multivectors: elements of $Lambda^k V$.
#[derive(Debug, Clone, Copy)]
pub struct Contravariant;
/// The variance of multiforms: elements of $Lambda^k V^*$.
#[derive(Debug, Clone, Copy)]
pub struct Covariant;

impl Variance for Contravariant {
  type Dual = Covariant;
  fn gramian(metric: &Metric) -> &Gramian {
    metric.vector_gramian()
  }
}
impl Variance for Covariant {
  type Dual = Contravariant;
  fn gramian(metric: &Metric) -> &Gramian {
    metric.covector_gramian()
  }
}

/// An element of $Lambda^k V$: a multivector.
pub type MultiVector = ExteriorElement<Contravariant>;
/// An element of $Lambda^k V^*$: a multiform.
pub type MultiForm = ExteriorElement<Covariant>;

/// The exterior power functor $Lambda^k$ applied to a linear map.
///
/// Matrix of $Lambda^k A$ in the colexicographically ordered bases of
/// $k$-blades: the $k$-th compound matrix of all $k times k$ minors,
/// $(Lambda^k A)_(I J) = det A[I, J]$.
///
/// Functoriality $Lambda^k (A B) = (Lambda^k A)(Lambda^k B)$ is the
/// Cauchy-Binet formula.
pub fn exterior_power(linear_map: &Matrix, grade: ExteriorGrade) -> Matrix {
  let nrows = exterior_dim(linear_map.nrows(), grade);
  let ncols = exterior_dim(linear_map.ncols(), grade);

  let mut power = Matrix::zeros(nrows, ncols);
  let mut minor = Matrix::zeros(grade, grade);
  for (i, row_basis) in exterior_bases(linear_map.nrows(), grade).enumerate() {
    for (j, col_basis) in exterior_bases(linear_map.ncols(), grade).enumerate() {
      for (ii, row) in row_basis.iter().enumerate() {
        for (jj, col) in col_basis.iter().enumerate() {
          minor[(ii, jj)] = linear_map[(row, col)];
        }
      }
      power[(i, j)] = minor.determinant();
    }
  }
  power
}

/// Construct Gramian on colexicographically ordered basis blades from the
/// Gramian on single elements.
///
/// The inner product on $Lambda^k$ is the exterior power of the inner product,
/// $inner(e_I, e_J)_(Lambda^k) = det [inner(e_i, e_j)]_(i in I, j in J)$.
pub fn multi_gramian(single_gramian: &Gramian, grade: ExteriorGrade) -> Gramian {
  Gramian::new_unchecked(exterior_power(single_gramian.matrix(), grade))
}

/// The induced inner product on multivectors $Lambda^k V$: $Lambda^k g$.
///
/// The variance-correct counterpart of [`multiform_gramian`]; the single
/// source of truth for which metric Gramian measures multivectors.
pub fn multivector_gramian(metric: &Metric, grade: ExteriorGrade) -> Gramian {
  multi_gramian(metric.vector_gramian(), grade)
}

/// The induced inner product on multiforms $Lambda^k V^*$: $Lambda^k g^(-1)$.
///
/// The variance-correct counterpart of [`multivector_gramian`]; the single
/// source of truth for which metric Gramian measures multiforms.
pub fn multiform_gramian(metric: &Metric, grade: ExteriorGrade) -> Gramian {
  multi_gramian(metric.covector_gramian(), grade)
}

/// An element of an exterior algebra, of the given [`Variance`].
///
/// Coefficients on the colexicographically ordered basis blades.
#[derive(Debug, Clone)]
pub struct ExteriorElement<V: Variance> {
  coeffs: Vector,
  dim: Dim,
  grade: ExteriorGrade,
  variance: PhantomData<V>,
}

impl<V: Variance> ExteriorElement<V> {
  pub fn new(coeffs: Vector, dim: Dim, grade: ExteriorGrade) -> Self {
    assert_eq!(coeffs.len(), exterior_dim(dim, grade));
    Self {
      coeffs,
      dim,
      grade,
      variance: PhantomData,
    }
  }

  pub fn scalar(v: f64, dim: Dim) -> Self {
    Self::new(na::dvector![v], dim, 0)
  }
  pub fn line(coeffs: Vector) -> Self {
    let dim = coeffs.len();
    Self::new(coeffs, dim, 1)
  }

  pub fn zero(dim: Dim, grade: ExteriorGrade) -> Self {
    Self::new(Vector::zeros(exterior_dim(dim, grade)), dim, grade)
  }
  pub fn one(dim: Dim) -> Self {
    Self::scalar(1.0, dim)
  }

  /// A single basis blade with the given sign.
  pub fn from_blade_signed(dim: Dim, sign: Sign, blade: Blade) -> Self {
    let mut element = Self::zero(dim, blade.card());
    element[blade] = sign.as_f64();
    element
  }

  pub fn into_grade1(self) -> Vector {
    assert_eq!(self.grade, 1);
    self.coeffs
  }

  pub fn dim(&self) -> Dim {
    self.dim
  }
  pub fn grade(&self) -> ExteriorGrade {
    self.grade
  }
  pub fn coeffs(&self) -> &Vector {
    &self.coeffs
  }
  pub fn into_coeffs(self) -> Vector {
    self.coeffs
  }

  pub fn basis_iter(&self) -> impl Iterator<Item = (f64, Blade)> + '_ {
    self
      .coeffs
      .iter()
      .copied()
      .zip(exterior_bases(self.dim, self.grade))
  }

  pub fn wedge(&self, other: &Self) -> Self {
    assert_eq!(self.dim, other.dim);
    let dim = self.dim;

    let new_grade = self.grade + other.grade;
    assert!(new_grade <= dim, "Wedge grade exceeds dimension.");

    let mut new_coeffs = Vector::zeros(exterior_dim(dim, new_grade));

    for (self_coeff, self_blade) in self.basis_iter() {
      if self_coeff == 0.0 {
        continue;
      }
      for (other_coeff, other_blade) in other.basis_iter() {
        if let Some((sign, merged)) = self_blade.union_signed(other_blade) {
          new_coeffs[merged.rank()] += sign.as_f64() * self_coeff * other_coeff;
        }
      }
    }

    Self::new(new_coeffs, dim, new_grade)
  }

  pub fn wedge_big(factors: impl IntoIterator<Item = Self>) -> Option<Self> {
    let mut factors = factors.into_iter();
    let first = factors.next()?;
    let prod = factors.fold(first, |acc, factor| acc.wedge(&factor));
    Some(prod)
  }

  /// The metric-free duality pairing with an element of the dual variance,
  /// $angle.l dot, dot angle.r: Lambda^k V^* times Lambda^k V -> RR$.
  pub fn pairing(&self, dual: &ExteriorElement<V::Dual>) -> f64 {
    assert!(self.dim == dual.dim && self.grade == dual.grade);
    self.coeffs.dot(&dual.coeffs)
  }

  /// The interior product (contraction)
  /// $iota_v: Lambda^k -> Lambda^(k-1)$ with a grade-1 element of the dual
  /// variance: a form contracted by a vector, a multivector by a covector.
  ///
  /// Metric-free. An antiderivation of degree -1 with $iota_v^2 = 0$: the
  /// dual of the wedge. On blades it is the alternating-deletion pattern of
  /// the simplicial boundary -- with the all-ones vector it IS the boundary
  /// operator, $diff = iota_bb(1)$.
  pub fn interior_product(&self, dual: &ExteriorElement<V::Dual>) -> Self {
    assert_eq!(dual.dim, self.dim);
    assert_eq!(dual.grade, 1, "Contraction is with a grade-1 element.");
    assert!(self.grade >= 1, "Interior product needs grade >= 1.");

    let mut contraction = Self::zero(self.dim, self.grade - 1);
    for (coeff, blade) in self.basis_iter() {
      if coeff == 0.0 {
        continue;
      }
      for (sign, index, deleted) in blade.deletions() {
        contraction.coeffs[deleted.rank()] += sign.as_f64() * dual.coeffs[index] * coeff;
      }
    }
    contraction
  }

  /// The inner product with the variance-appropriate metric Gramian:
  /// $Lambda^k g$ for multivectors, $Lambda^k g^(-1)$ for multiforms.
  /// Indefinite whenever the metric is: the sign is information, not error.
  pub fn inner(&self, other: &Self, metric: &Metric) -> f64 {
    assert!(self.dim == other.dim && self.grade == other.grade);
    multi_gramian(V::gramian(metric), self.grade).inner(&self.coeffs, &other.coeffs)
  }
  /// Magnitude $sqrt(abs(inner(v, v)))$. On an indefinite metric the sign of
  /// [`Self::inner`] carries the causal character separately; this is never
  /// NaN.
  pub fn norm(&self, metric: &Metric) -> f64 {
    self.inner(self, metric).abs().sqrt()
  }

  /// The Hodge star $star: Lambda^k -> Lambda^(n-k)$, defined by
  /// $alpha wedge star beta = inner(alpha, beta) vol$ for all $alpha$,
  /// where $vol$ is the unit-volume element of this variance, with density
  /// $sqrt(abs(det g))$ on any signature.
  ///
  /// The involution is $star star = (-1)^(k(n-k)) (-1)^q$ with $q$ the number
  /// of negative eigenvalues of the metric, $(-1)^q = sgn(det g)$: the one
  /// fact separating Riemannian from Lorentzian Hodge theory, read off the
  /// metric's own signature rather than assumed.
  ///
  /// Assumes the positively oriented standard basis. For the Euclidean metric
  /// the star is exactly the signed complement of each basis blade
  /// ([`Combination::complement_signed`]).
  pub fn hodge_star(&self, metric: &Metric) -> Self {
    let dim = self.dim;
    assert_eq!(metric.dim(), dim);

    let gramian = V::gramian(metric);
    let weighted = (exterior_power(gramian.matrix(), self.grade) * &self.coeffs)
      / multi_gramian(gramian, dim).det_sqrt();

    let mut star = Self::zero(dim, dim - self.grade);
    for (blade, &coeff) in exterior_bases(dim, self.grade).zip(weighted.iter()) {
      let (sign, complement) = blade.complement_signed(dim);
      star.coeffs[complement.rank()] = sign.as_f64() * coeff;
    }
    star
  }

  pub fn eq_epsilon(&self, other: &Self, eps: f64) -> bool {
    self.dim == other.dim
      && self.grade == other.grade
      && (&self.coeffs - &other.coeffs).norm_squared() <= eps.powi(2)
  }
}

impl MultiVector {
  /// Pushforward along a linear map $A: V -> W$: the covariant action of
  /// the exterior power functor, $Lambda^k A$ on coefficients.
  ///
  /// Adjoint to [`MultiForm::pullback`] under the duality [`pairing`]:
  /// $angle.l A^* omega, v angle.r = angle.l omega, A_* v angle.r$.
  ///
  /// [`pairing`]: ExteriorElement::pairing
  pub fn pushforward(&self, linear_map: &Matrix) -> MultiVector {
    assert_eq!(self.dim, linear_map.ncols());
    let coeffs = exterior_power(linear_map, self.grade) * &self.coeffs;
    Self::new(coeffs, linear_map.nrows(), self.grade)
  }

  /// The musical isomorphism $flat: Lambda^k V -> Lambda^k V^*$,
  /// lowering indices with $Lambda^k g$.
  pub fn flat(&self, metric: &Metric) -> MultiForm {
    let coeffs = exterior_power(metric.vector_gramian().matrix(), self.grade) * &self.coeffs;
    MultiForm::new(coeffs, self.dim, self.grade)
  }
}

impl MultiForm {
  /// Pullback along a linear map $A: V -> W$: the contravariant action of
  /// the exterior power functor, $(Lambda^k A)^T$ on coefficients.
  ///
  /// Adjoint to [`MultiVector::pushforward`] under the duality [`pairing`]:
  /// $angle.l A^* omega, v angle.r = angle.l omega, A_* v angle.r$.
  ///
  /// [`pairing`]: ExteriorElement::pairing
  pub fn pullback(&self, linear_map: &Matrix) -> MultiForm {
    assert_eq!(self.dim, linear_map.nrows());
    let coeffs = exterior_power(linear_map, self.grade).transpose() * &self.coeffs;
    Self::new(coeffs, linear_map.ncols(), self.grade)
  }

  /// The musical isomorphism $sharp: Lambda^k V^* -> Lambda^k V$,
  /// raising indices with $Lambda^k g^(-1)$. Inverse of
  /// [`MultiVector::flat`].
  pub fn sharp(&self, metric: &Metric) -> MultiVector {
    let coeffs = exterior_power(metric.covector_gramian().matrix(), self.grade) * &self.coeffs;
    MultiVector::new(coeffs, self.dim, self.grade)
  }
}

impl<V: Variance> std::ops::Add for ExteriorElement<V> {
  type Output = Self;
  fn add(mut self, other: Self) -> Self::Output {
    self += other;
    self
  }
}
impl<V: Variance> std::ops::AddAssign for ExteriorElement<V> {
  fn add_assign(&mut self, other: Self) {
    assert_eq!(self.dim, other.dim);
    assert_eq!(self.grade, other.grade);
    self.coeffs += other.coeffs;
  }
}

impl<V: Variance> std::ops::Sub for ExteriorElement<V> {
  type Output = Self;
  fn sub(mut self, other: Self) -> Self::Output {
    self -= other;
    self
  }
}
impl<V: Variance> std::ops::SubAssign for ExteriorElement<V> {
  fn sub_assign(&mut self, other: Self) {
    assert_eq!(self.dim, other.dim);
    assert_eq!(self.grade, other.grade);
    self.coeffs -= other.coeffs;
  }
}

impl<V: Variance> std::ops::Mul<f64> for ExteriorElement<V> {
  type Output = Self;
  fn mul(mut self, scalar: f64) -> Self::Output {
    self *= scalar;
    self
  }
}
impl<V: Variance> std::ops::MulAssign<f64> for ExteriorElement<V> {
  fn mul_assign(&mut self, scalar: f64) {
    self.coeffs *= scalar;
  }
}
impl<V: Variance> std::ops::Mul<ExteriorElement<V>> for f64 {
  type Output = ExteriorElement<V>;
  fn mul(self, rhs: ExteriorElement<V>) -> Self::Output {
    rhs * self
  }
}

impl<V: Variance> std::ops::Index<Blade> for ExteriorElement<V> {
  type Output = f64;
  fn index(&self, blade: Blade) -> &Self::Output {
    assert_eq!(blade.card(), self.grade);
    assert!(blade.iter().all(|i| i < self.dim));
    &self.coeffs[blade.rank()]
  }
}
impl<V: Variance> std::ops::IndexMut<Blade> for ExteriorElement<V> {
  fn index_mut(&mut self, blade: Blade) -> &mut Self::Output {
    assert_eq!(blade.card(), self.grade);
    assert!(blade.iter().all(|i| i < self.dim));
    &mut self.coeffs[blade.rank()]
  }
}
impl<V: Variance> std::ops::Index<usize> for ExteriorElement<V> {
  type Output = f64;
  fn index(&self, index: usize) -> &Self::Output {
    &self.coeffs[index]
  }
}

impl<V: Variance> std::iter::Sum for ExteriorElement<V> {
  fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
    let mut iter = iter.into_iter();
    let mut sum = iter.next().unwrap();
    for element in iter {
      sum += element;
    }
    sum
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  use approx::assert_relative_eq;

  /// Deterministic full-rank-ish test matrix.
  fn test_matrix(nrows: usize, ncols: usize, seed: usize) -> Matrix {
    Matrix::from_fn(nrows, ncols, |i, j| {
      ((seed + 3 * i + 7 * j) % 5) as f64 / 5.0 + if i == j { 1.0 } else { 0.0 }
    })
  }
  fn test_element<V: Variance>(dim: Dim, grade: ExteriorGrade, seed: usize) -> ExteriorElement<V> {
    ExteriorElement::new(
      Vector::from_fn(exterior_dim(dim, grade), |i, _| {
        ((seed + 5 * i) % 7) as f64 - 3.0
      }),
      dim,
      grade,
    )
  }
  /// A non-trivial Riemannian metric.
  fn test_metric(dim: Dim) -> Metric {
    let a = test_matrix(dim, dim, 5);
    Metric::new(Gramian::new(a.transpose() * a + Matrix::identity(dim, dim)))
  }
  /// A non-diagonal metric of signature $(dim - q, q)$: the flat model pulled
  /// back along an invertible map, which preserves the signature (Sylvester)
  /// while filling in off-diagonal entries.
  fn test_pseudo_metric(dim: Dim, q: usize) -> Metric {
    let j = Matrix::from_fn(dim, dim, |i, jj| {
      if i == jj {
        1.0
      } else if i > jj {
        ((3 * i + 5 * jj) % 4) as f64 / 8.0
      } else {
        0.0
      }
    });
    Metric::new(Gramian::pseudo_euclidean(dim - q, q).pullback(&j))
  }

  /// Functoriality of the exterior power: the Cauchy-Binet formula
  /// $Lambda^k (A B) = (Lambda^k A)(Lambda^k B)$.
  #[test]
  fn exterior_power_functoriality() {
    for (n, m) in [(2, 2), (3, 3), (4, 4), (2, 3), (3, 2), (2, 4)] {
      let a = test_matrix(n, m, 1);
      let b = test_matrix(m, n, 2);
      for k in 0..=n.min(m) {
        let power_ab = exterior_power(&(&a * &b), k);
        let power_a_power_b = exterior_power(&a, k) * exterior_power(&b, k);
        assert_relative_eq!(power_ab, power_a_power_b, epsilon = 1e-12);
      }
    }
  }

  /// $Lambda^n A = det A$ and $Lambda^k id = id$.
  #[test]
  fn exterior_power_det_and_identity() {
    for n in 1..=4 {
      let a = test_matrix(n, n, 3);
      assert_relative_eq!(
        exterior_power(&a, n)[(0, 0)],
        a.determinant(),
        epsilon = 1e-12
      );
      for k in 0..=n {
        let id = Matrix::identity(n, n);
        assert_relative_eq!(
          exterior_power(&id, k),
          Matrix::identity(exterior_dim(n, k), exterior_dim(n, k))
        );
      }
    }
  }

  /// Pushforward and pullback are adjoint under the duality pairing:
  /// $angle.l A^* omega, v angle.r = angle.l omega, A_* v angle.r$.
  #[test]
  fn pullback_pushforward_adjoint() {
    for (n, m) in [(2, 2), (3, 3), (3, 2), (4, 3)] {
      let a = test_matrix(n, m, 4);
      for k in 0..=n.min(m) {
        let form: MultiForm = test_element(n, k, 1);
        let vector: MultiVector = test_element(m, k, 2);
        let lhs = form.pullback(&a).pairing(&vector);
        let rhs = form.pairing(&vector.pushforward(&a));
        assert_relative_eq!(lhs, rhs, epsilon = 1e-12);
      }
    }
  }

  /// The pullback is the same as wedging the pulled-back constituent 1-forms.
  #[test]
  fn pullback_is_wedge_of_pullbacks() {
    for (n, m) in [(2, 2), (3, 3), (3, 2), (4, 3)] {
      let a = test_matrix(n, m, 4);
      for k in 0..=n.min(m) {
        let form: MultiForm = test_element(n, k, 6);
        let computed = form.pullback(&a);
        let expected: MultiForm = form
          .basis_iter()
          .map(|(coeff, blade)| {
            coeff
              * MultiForm::wedge_big(blade.iter().map(|i| MultiForm::line(a.row(i).transpose())))
                .unwrap_or(MultiForm::one(m))
          })
          .sum();
        assert_relative_eq!(computed.coeffs(), expected.coeffs(), epsilon = 1e-12);
      }
    }
  }

  /// The musical isomorphisms are inverse and turn the duality pairing into
  /// the metric inner product: $angle.l v^flat, w angle.r = inner(v, w)_g$.
  #[test]
  fn musical_isomorphisms() {
    for dim in 1..=4 {
      let metric = test_metric(dim);
      for grade in 0..=dim {
        let v: MultiVector = test_element(dim, grade, 1);
        let w: MultiVector = test_element(dim, grade, 2);

        let roundtrip = v.flat(&metric).sharp(&metric);
        assert_relative_eq!(roundtrip.coeffs(), v.coeffs(), epsilon = 1e-12);

        assert_relative_eq!(
          v.flat(&metric).pairing(&w),
          v.inner(&w, &metric),
          epsilon = 1e-12
        );
        // Sharp is the adjoint direction: <omega, alpha#> = <omega, alpha>_{g^-1}.
        let omega: MultiForm = test_element(dim, grade, 3);
        let alpha: MultiForm = test_element(dim, grade, 4);
        assert_relative_eq!(
          omega.pairing(&alpha.sharp(&metric)),
          omega.inner(&alpha, &metric),
          epsilon = 1e-12
        );
      }
    }
  }

  /// For the Euclidean metric the Hodge star is the signed complement of
  /// each basis blade.
  #[test]
  fn hodge_star_euclidean_is_signed_complement() {
    for dim in 1..=4 {
      let euclidean = Metric::standard(dim);
      for grade in 0..=dim {
        for blade in exterior_bases(dim, grade) {
          let element = MultiForm::from_blade_signed(dim, Sign::Pos, blade);
          let star = element.hodge_star(&euclidean);
          let (sign, complement) = blade.complement_signed(dim);
          let expected = MultiForm::from_blade_signed(dim, sign, complement);
          assert_relative_eq!(star.coeffs(), expected.coeffs());
        }
      }
    }
  }

  /// $star star = (-1)^(k(n-k)) (-1)^q$ on any metric of signature
  /// $(n - q, q)$, for both variances: swept over dimension, grade *and*
  /// signature, the Riemannian $q = 0$ one case among all.
  #[test]
  fn hodge_star_involution() {
    for dim in 1..=4 {
      for q in 0..=dim {
        for metric in [
          Metric::new(Gramian::pseudo_euclidean(dim - q, q)),
          test_pseudo_metric(dim, q),
        ] {
          for grade in 0..=dim {
            let sign = Sign::from_parity(grade * (dim - grade)) * Sign::from_parity(q);

            let form: MultiForm = test_element(dim, grade, 2);
            let star_star = form.hodge_star(&metric).hodge_star(&metric);
            assert_relative_eq!(
              star_star.coeffs(),
              &(sign.as_f64() * form).coeffs(),
              epsilon = 1e-12
            );

            let vector: MultiVector = test_element(dim, grade, 3);
            let star_star = vector.hodge_star(&metric).hodge_star(&metric);
            assert_relative_eq!(
              star_star.coeffs(),
              &(sign.as_f64() * vector).coeffs(),
              epsilon = 1e-12
            );
          }
        }
      }
    }
  }

  /// The defining property $alpha wedge star beta = inner(alpha, beta) vol$,
  /// tying together wedge, inner and hodge_star: on every signature, with the
  /// indefinite inner product and the volume density $sqrt(abs(det g))$.
  #[test]
  fn wedge_with_star_is_inner_times_volume() {
    for dim in 1..=4 {
      for q in 0..=dim {
        for metric in [
          Metric::new(Gramian::pseudo_euclidean(dim - q, q)),
          test_pseudo_metric(dim, q),
        ] {
          for grade in 0..=dim {
            let alpha: MultiForm = test_element(dim, grade, 3);
            let beta: MultiForm = test_element(dim, grade, 4);

            let wedge = alpha.wedge(&beta.hodge_star(&metric));

            let inner = alpha.inner(&beta, &metric);
            // The volume form has coefficient sqrt(|det g|).
            let volume = metric.det_sqrt();

            assert_eq!(wedge.grade(), dim);
            assert_relative_eq!(wedge[0], inner * volume, epsilon = 1e-12);
          }
        }
      }
    }
  }

  /// The Clifford relation: the Clifford action of a covector $a$ on forms,
  /// $c_a = a wedge dot - iota_(a^sharp)$, squares to a scalar,
  /// $c_a c_a = -inner(a, a)_(g^(-1))$ -- on every signature. This is the
  /// algebraic identity behind the Hodge--Dirac operator: for a plane wave
  /// $u = sin(a dot x) thin omega$ on flat space, $(dif + delta) u =
  /// cos(a dot x) thin c_a omega$, so $c_a^2$ scalar is exactly
  /// $sans(D)^2 = Delta$ acting as $inner(a, a)$ -- the dispersion relation,
  /// null covectors giving massless waves.
  #[test]
  fn clifford_relation() {
    for dim in 1..=4 {
      for q in 0..=dim {
        let metric = test_pseudo_metric(dim, q);
        let a = MultiForm::line(Vector::from_fn(dim, |i, _| (i as f64) - 1.5));
        let a_sharp = a.sharp(&metric);
        let a_norm_sq = a.inner(&a, &metric);

        for grade in 0..=dim {
          let omega: MultiForm = test_element(dim, grade, 4);

          // c_a c_a omega, collecting the four terms by their grades:
          // (a wedge)^2 = 0 and iota^2 = 0, so only the two cross terms
          // survive, both landing back at `grade`.
          let wedge_then_contract = if grade < dim {
            a.wedge(&omega).interior_product(&a_sharp)
          } else {
            MultiForm::zero(dim, grade)
          };
          let contract_then_wedge = if grade >= 1 {
            a.wedge(&omega.interior_product(&a_sharp))
          } else {
            MultiForm::zero(dim, grade)
          };
          let squared = -1.0 * wedge_then_contract - contract_then_wedge;

          let expected = -a_norm_sq * omega;
          assert_relative_eq!(squared.coeffs(), expected.coeffs(), epsilon = 1e-9);
        }
      }
    }
  }

  /// The Lorentzian Hodge star on flat Minkowski space (mostly-plus,
  /// $eta = "diag"(-1, +1, dots.c, +1)$), checked against the closed form on
  /// basis blades: $star dif x^I = (product_(i in I) eta^(i i)) space
  /// epsilon(I, I^c) space dif x^(I^c)$ -- each timelike factor in the blade
  /// flips the sign of the Euclidean star, nothing else changes.
  #[test]
  fn hodge_star_minkowski_closed_form() {
    for dim in 1..=4 {
      let eta = Metric::minkowski(dim);
      for grade in 0..=dim {
        for blade in exterior_bases(dim, grade) {
          let element = MultiForm::from_blade_signed(dim, Sign::Pos, blade);
          let star = element.hodge_star(&eta);

          // One sign flip per timelike ($eta^(0 0) = -1$) factor of the blade.
          let eta_sign = Sign::from_parity(blade.contains(0) as usize);
          let (complement_sign, complement) = blade.complement_signed(dim);
          let expected = MultiForm::from_blade_signed(dim, eta_sign * complement_sign, complement);
          assert_relative_eq!(star.coeffs(), expected.coeffs());
        }
      }
    }
  }

  /// $iota_v$ is an antiderivation:
  /// $iota_v (alpha wedge beta) = (iota_v alpha) wedge beta
  ///   + (-1)^k alpha wedge (iota_v beta)$.
  #[test]
  fn interior_product_antiderivation() {
    for dim in 2..=4 {
      let vector = MultiVector::line(Vector::from_fn(dim, |i, _| (i + 1) as f64));
      for grade_a in 1..dim {
        for grade_b in 1..=(dim - grade_a) {
          let alpha: MultiForm = test_element(dim, grade_a, 5);
          let beta: MultiForm = test_element(dim, grade_b, 6);

          let lhs = alpha.wedge(&beta).interior_product(&vector);
          let sign = Sign::from_parity(grade_a);
          let rhs = alpha.interior_product(&vector).wedge(&beta)
            + sign.as_f64() * alpha.wedge(&beta.interior_product(&vector));
          assert_relative_eq!(lhs.coeffs(), rhs.coeffs(), epsilon = 1e-12);
        }
      }
    }
  }

  /// $iota_v compose iota_v = 0$ -- the same law as $diff compose diff = 0$.
  #[test]
  fn interior_product_squares_to_zero() {
    for dim in 2..=4 {
      let vector = MultiVector::line(Vector::from_fn(dim, |i, _| (2 * i + 1) as f64));
      for grade in 2..=dim {
        let element: MultiForm = test_element(dim, grade, 7);
        let twice = element.interior_product(&vector).interior_product(&vector);
        assert_relative_eq!(twice.coeffs().norm(), 0.0);
      }
    }
  }

  /// The interior product is adjoint to wedging with the flat of the vector:
  /// $inner(iota_v alpha, beta) = inner(alpha, v^flat wedge beta)$ (Euclidean).
  #[test]
  fn interior_product_adjoint_to_wedge() {
    for dim in 2..=4 {
      let euclidean = Metric::standard(dim);
      let vector = MultiVector::line(Vector::from_fn(dim, |i, _| (i + 2) as f64));
      let vector_flat = vector.flat(&euclidean);
      for grade in 1..=dim {
        let alpha: MultiForm = test_element(dim, grade, 8);
        let beta: MultiForm = test_element(dim, grade - 1, 9);

        let lhs = alpha.interior_product(&vector).inner(&beta, &euclidean);
        let rhs = alpha.inner(&vector_flat.wedge(&beta), &euclidean);
        assert_relative_eq!(lhs, rhs, epsilon = 1e-12);
      }
    }
  }

  #[test]
  fn compute_wedge() {
    let a = MultiForm::new(na::dvector![1.0, 0.0, 0.0], 3, 1);
    let b = MultiForm::new(na::dvector![0.0, 1.0, 0.0], 3, 1);

    let computed_ab = a.wedge(&b);
    let expected_ab = MultiForm::from_blade_signed(3, Sign::Pos, Blade::from_increasing([0, 1]));
    assert_eq!(computed_ab.coeffs, expected_ab.coeffs);
  }

  #[test]
  fn wedge_antisymmetry() {
    let form_a = MultiForm::new(na::dvector![1.0, 0.0, 0.0], 3, 1);
    let form_b = MultiForm::new(na::dvector![0.0, 1.0, 0.0], 3, 1);

    let ab = form_a.wedge(&form_b);
    let ba = form_b.wedge(&form_a);
    assert_eq!(ab.coeffs, -ba.coeffs);
  }

  /// Signed sums of arbitrarily ordered wedge words canonicalize correctly.
  #[test]
  fn canonical_conversion() {
    let dim = 4;
    let mut e0 = MultiForm::zero(dim, 3);
    for (coeff, word) in [
      (1.0, vec![2, 0, 1]),
      (3.0, vec![1, 3, 2]),
      (-2.0, vec![0, 2, 1]),
      (3.0, vec![0, 1, 2]),
    ] {
      let (sign, blade) = Blade::from_word(word).unwrap();
      e0[blade] += sign.as_f64() * coeff;
    }

    let mut e1 = MultiForm::zero(dim, 3);
    e1[Blade::from_increasing([0, 1, 2])] = 6.0;
    e1[Blade::from_increasing([1, 2, 3])] = -3.0;

    assert!(e0.eq_epsilon(&e1, 1e-12));
  }

  #[test]
  fn multi_gramian_euclidean() {
    for n in 0..=3 {
      let gramian = Gramian::standard(n);
      for k in 0..=n {
        let expected_gram = Gramian::standard(binomial(n, k));
        let computed_gram = multi_gramian(&gramian, k);
        assert_relative_eq!(computed_gram.matrix(), expected_gram.matrix());
      }
    }
  }
}
