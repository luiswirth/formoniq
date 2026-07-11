extern crate nalgebra as na;

pub mod field;
pub mod list;

use common::{
  combo::{binomial, combinations, Combination, Sign},
  gramian::Gramian,
  linalg::nalgebra::{Matrix, Vector},
};

pub type Dim = usize;
pub type ExteriorGrade = usize;

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

/// The exterior power functor $Lambda^k$ applied to a linear map.
///
/// Matrix of $Lambda^k A$ in the colexicographically ordered bases of
/// $k$-blades: the $k$-th compound matrix of all $k times k$ minors,
/// $(Lambda^k A)_(I J) = det A[I, J]$.
///
/// Functoriality $Lambda^k (A B) = (Lambda^k A)(Lambda^k B)$ is the
/// Cauchy-Binet formula. This single primitive induces the inner product on
/// $Lambda^k$ (multi_gramian), the pullback of $k$-forms (precompose_form)
/// and the coefficients of a wedge of vectors (spanning multivector).
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

/// An element of an exterior algebra.
///
/// Coefficients on the colexicographically ordered basis blades.
#[derive(Debug, Clone)]
pub struct ExteriorElement {
  coeffs: Vector,
  dim: Dim,
  grade: ExteriorGrade,
}

impl ExteriorElement {
  pub fn new(coeffs: Vector, dim: Dim, grade: ExteriorGrade) -> Self {
    assert_eq!(coeffs.len(), exterior_dim(dim, grade));
    Self { coeffs, dim, grade }
  }

  pub fn scalar(v: f64, dim: Dim) -> ExteriorElement {
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
    assert!(self.grade == 1);
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

  /// The interior product (contraction) $iota_v: Lambda^k -> Lambda^(k-1)$
  /// with a vector.
  ///
  /// Metric-free. An antiderivation of degree -1 with $iota_v^2 = 0$: the
  /// dual of the wedge. On blades it is the alternating-deletion pattern of
  /// the simplicial boundary, weighted by the vector components -- with the
  /// all-ones vector it IS the boundary operator, $diff = iota_bb(1)$.
  ///
  /// Read as: a k-form contracted by a tangent vector, or equally a
  /// k-vector contracted by a covector.
  pub fn interior_product(&self, vector: &Vector) -> Self {
    assert_eq!(vector.len(), self.dim);
    assert!(self.grade >= 1, "Interior product needs grade >= 1.");

    let mut contraction = Self::zero(self.dim, self.grade - 1);
    for (coeff, blade) in self.basis_iter() {
      if coeff == 0.0 {
        continue;
      }
      for (sign, index, deleted) in blade.deletions() {
        contraction.coeffs[deleted.rank()] += sign.as_f64() * vector[index] * coeff;
      }
    }
    contraction
  }

  pub fn eq_epsilon(&self, other: &Self, eps: f64) -> bool {
    self.dim == other.dim
      && self.grade == other.grade
      && (&self.coeffs - &other.coeffs).norm_squared() <= eps.powi(2)
  }
}

impl std::ops::Add<ExteriorElement> for ExteriorElement {
  type Output = Self;
  fn add(mut self, other: ExteriorElement) -> Self::Output {
    self += other;
    self
  }
}
impl std::ops::AddAssign<ExteriorElement> for ExteriorElement {
  fn add_assign(&mut self, other: ExteriorElement) {
    assert_eq!(self.dim, other.dim);
    assert_eq!(self.grade, other.grade);
    self.coeffs += other.coeffs;
  }
}

impl std::ops::Sub<ExteriorElement> for ExteriorElement {
  type Output = Self;
  fn sub(mut self, other: ExteriorElement) -> Self::Output {
    self -= other;
    self
  }
}
impl std::ops::SubAssign<ExteriorElement> for ExteriorElement {
  fn sub_assign(&mut self, other: ExteriorElement) {
    assert_eq!(self.dim, other.dim);
    assert_eq!(self.grade, other.grade);
    self.coeffs -= other.coeffs;
  }
}

impl std::ops::Mul<f64> for ExteriorElement {
  type Output = Self;
  fn mul(mut self, scalar: f64) -> Self::Output {
    self *= scalar;
    self
  }
}
impl std::ops::MulAssign<f64> for ExteriorElement {
  fn mul_assign(&mut self, scalar: f64) {
    self.coeffs *= scalar;
  }
}
impl std::ops::Mul<ExteriorElement> for f64 {
  type Output = ExteriorElement;
  fn mul(self, rhs: ExteriorElement) -> Self::Output {
    rhs * self
  }
}

impl std::ops::Index<Blade> for ExteriorElement {
  type Output = f64;
  fn index(&self, blade: Blade) -> &Self::Output {
    assert!(blade.card() == self.grade);
    assert!(blade.iter().all(|i| i < self.dim));
    &self.coeffs[blade.rank()]
  }
}
impl std::ops::IndexMut<Blade> for ExteriorElement {
  fn index_mut(&mut self, blade: Blade) -> &mut Self::Output {
    assert!(blade.card() == self.grade);
    assert!(blade.iter().all(|i| i < self.dim));
    &mut self.coeffs[blade.rank()]
  }
}
impl std::ops::Index<usize> for ExteriorElement {
  type Output = f64;
  fn index(&self, index: usize) -> &Self::Output {
    &self.coeffs[index]
  }
}

impl std::iter::Sum for ExteriorElement {
  fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
    let mut iter = iter.into_iter();
    let mut sum = iter.next().unwrap();
    for element in iter {
      sum += element;
    }
    sum
  }
}

pub type MultiVector = ExteriorElement;
pub type MultiForm = ExteriorElement;
impl MultiForm {
  /// Precompose k-form by some linear map: the pullback of differential
  /// k-forms.
  ///
  /// The contravariant functor $Lambda^k$ acting on coefficients,
  /// $c mapsto (Lambda^k A)^T c$.
  pub fn precompose_form(&self, linear_map: &Matrix) -> Self {
    assert_eq!(self.dim, linear_map.nrows());
    let coeffs = exterior_power(linear_map, self.grade).transpose() * &self.coeffs;
    Self::new(coeffs, linear_map.ncols(), self.grade)
  }

  pub fn apply_form_to_vector(&self, vector: &MultiVector) -> f64 {
    assert!(self.dim == vector.dim && self.grade == vector.grade);
    self.coeffs.dot(&vector.coeffs)
  }

  /// The Hodge star $star: Lambda^k -> Lambda^(n-k)$, defined by
  /// $alpha wedge star beta = inner(alpha, beta)_g vol_g$ for all $alpha$.
  ///
  /// Metric-dependent: takes the metric tensor $g$ on tangent vectors.
  /// The inner product on k-forms is $Lambda^k g^(-1)$ and the volume form
  /// is $sqrt(det g) dif x^(1 dots n)$; the positively oriented standard
  /// basis is assumed. In coefficients,
  /// $(star beta)_(I^c) = sqrt(det g) med epsilon(I, I^c) (Lambda^k g^(-1) beta)_I$,
  /// so for the Euclidean metric the star is exactly the signed complement
  /// of each basis blade ([`Combination::complement_signed`]).
  pub fn hodge_star(&self, metric: &Gramian) -> Self {
    let dim = self.dim;
    assert_eq!(metric.dim(), dim);

    let form_gramian = multi_gramian(&metric.clone().inverse(), self.grade);
    let weighted = metric.det_sqrt() * (form_gramian.matrix() * &self.coeffs);

    let mut star = Self::zero(dim, dim - self.grade);
    for (blade, &coeff) in exterior_bases(dim, self.grade).zip(weighted.iter()) {
      let (sign, complement) = blade.complement_signed(dim);
      star.coeffs[complement.rank()] = sign.as_f64() * coeff;
    }
    star
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
      assert_relative_eq!(exterior_power(&a, n)[(0, 0)], a.determinant(), epsilon = 1e-12);
      for k in 0..=n {
        let id = Matrix::identity(n, n);
        assert_relative_eq!(
          exterior_power(&id, k),
          Matrix::identity(exterior_dim(n, k), exterior_dim(n, k))
        );
      }
    }
  }

  /// The pullback is the same as wedging the pulled-back constituent 1-forms.
  #[test]
  fn precompose_is_wedge_of_pullbacks() {
    for (n, m) in [(2, 2), (3, 3), (3, 2), (4, 3)] {
      let a = test_matrix(n, m, 4);
      for k in 0..=n.min(m) {
        let form = ExteriorElement::new(
          Vector::from_fn(exterior_dim(n, k), |i, _| (i + 1) as f64),
          n,
          k,
        );
        let computed = form.precompose_form(&a);
        let expected: ExteriorElement = form
          .basis_iter()
          .map(|(coeff, blade)| {
            coeff
              * MultiForm::wedge_big(blade.iter().map(|i| MultiForm::line(a.row(i).transpose())))
                .unwrap_or(ExteriorElement::one(m))
          })
          .sum();
        assert_relative_eq!(computed.coeffs(), expected.coeffs(), epsilon = 1e-12);
      }
    }
  }

  fn test_element(dim: Dim, grade: ExteriorGrade, seed: usize) -> ExteriorElement {
    ExteriorElement::new(
      Vector::from_fn(exterior_dim(dim, grade), |i, _| {
        ((seed + 5 * i) % 7) as f64 - 3.0
      }),
      dim,
      grade,
    )
  }
  /// A non-trivial s.p.d. metric tensor.
  fn test_metric(dim: Dim) -> Gramian {
    let a = test_matrix(dim, dim, 5);
    Gramian::new(a.transpose() * a + Matrix::identity(dim, dim))
  }

  /// For the Euclidean metric the Hodge star is the signed complement of
  /// each basis blade.
  #[test]
  fn hodge_star_euclidean_is_signed_complement() {
    for dim in 1..=4 {
      let euclidean = Gramian::standard(dim);
      for grade in 0..=dim {
        for blade in exterior_bases(dim, grade) {
          let element = ExteriorElement::from_blade_signed(dim, Sign::Pos, blade);
          let star = element.hodge_star(&euclidean);
          let (sign, complement) = blade.complement_signed(dim);
          let expected = ExteriorElement::from_blade_signed(dim, sign, complement);
          assert_relative_eq!(star.coeffs(), expected.coeffs());
        }
      }
    }
  }

  /// $star star = (-1)^(k(n-k))$ on any Riemannian metric.
  #[test]
  fn hodge_star_involution() {
    for dim in 1..=4 {
      for metric in [Gramian::standard(dim), test_metric(dim)] {
        for grade in 0..=dim {
          let element = test_element(dim, grade, 2);
          let star_star = element.hodge_star(&metric).hodge_star(&metric);
          let sign = Sign::from_parity(grade * (dim - grade));
          assert_relative_eq!(
            star_star.coeffs(),
            &(sign.as_f64() * element).coeffs(),
            epsilon = 1e-12
          );
        }
      }
    }
  }

  /// The defining property $alpha wedge star beta = inner(alpha, beta)_g vol_g$,
  /// tying together wedge, multi_gramian and hodge_star.
  #[test]
  fn wedge_with_star_is_inner_times_volume() {
    for dim in 1..=4 {
      for metric in [Gramian::standard(dim), test_metric(dim)] {
        for grade in 0..=dim {
          let alpha = test_element(dim, grade, 3);
          let beta = test_element(dim, grade, 4);

          let wedge = alpha.wedge(&beta.hodge_star(&metric));

          let inner = multi_gramian(&metric.clone().inverse(), grade)
            .inner(alpha.coeffs(), beta.coeffs());
          let volume = metric.det_sqrt();

          assert_eq!(wedge.grade(), dim);
          assert_relative_eq!(wedge[0], inner * volume, epsilon = 1e-12);
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
      let vector = Vector::from_fn(dim, |i, _| (i + 1) as f64);
      for grade_a in 1..dim {
        for grade_b in 1..=(dim - grade_a) {
          let alpha = test_element(dim, grade_a, 5);
          let beta = test_element(dim, grade_b, 6);

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
      let vector = Vector::from_fn(dim, |i, _| (2 * i + 1) as f64);
      for grade in 2..=dim {
        let element = test_element(dim, grade, 7);
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
      let vector = Vector::from_fn(dim, |i, _| (i + 2) as f64);
      let vector_flat = ExteriorElement::line(vector.clone());
      for grade in 1..=dim {
        let alpha = test_element(dim, grade, 8);
        let beta = test_element(dim, grade - 1, 9);

        let lhs = alpha.interior_product(&vector).coeffs().dot(beta.coeffs());
        let rhs = alpha.coeffs().dot(vector_flat.wedge(&beta).coeffs());
        assert_relative_eq!(lhs, rhs, epsilon = 1e-12);
      }
    }
  }

  #[test]
  fn compute_wedge() {
    let a = ExteriorElement::new(na::dvector![1.0, 0.0, 0.0], 3, 1);
    let b = ExteriorElement::new(na::dvector![0.0, 1.0, 0.0], 3, 1);

    let computed_ab = a.wedge(&b);
    let expected_ab = ExteriorElement::from_blade_signed(
      3,
      Sign::Pos,
      Blade::from_increasing([0, 1]),
    );
    assert_eq!(computed_ab.coeffs, expected_ab.coeffs);
  }

  #[test]
  fn wedge_antisymmetry() {
    let form_a = ExteriorElement::new(na::dvector![1.0, 0.0, 0.0], 3, 1);
    let form_b = ExteriorElement::new(na::dvector![0.0, 1.0, 0.0], 3, 1);

    let ab = form_a.wedge(&form_b);
    let ba = form_b.wedge(&form_a);
    assert_eq!(ab.coeffs, -ba.coeffs);
  }

  #[test]
  fn wedge_with_zero() {
    let form = ExteriorElement::new(na::dvector![1.0, 0.0, 0.0], 3, 1);
    let zero_form = ExteriorElement::zero(3, 1);

    let wedge_computed = form.wedge(&zero_form);
    let wedge_expected = ExteriorElement::zero(3, 2);
    assert_eq!(wedge_computed.coeffs, wedge_expected.coeffs);
  }

  /// Signed sums of arbitrarily ordered wedge words canonicalize correctly.
  #[test]
  fn canonical_conversion() {
    let dim = 4;
    let mut e0 = ExteriorElement::zero(dim, 3);
    for (coeff, word) in [
      (1.0, vec![2, 0, 1]),
      (3.0, vec![1, 3, 2]),
      (-2.0, vec![0, 2, 1]),
      (3.0, vec![0, 1, 2]),
    ] {
      let (sign, blade) = Blade::from_word(word).unwrap();
      e0[blade] += sign.as_f64() * coeff;
    }

    let mut e1 = ExteriorElement::zero(dim, 3);
    e1[Blade::from_increasing([0, 1, 2])] = 6.0;
    e1[Blade::from_increasing([1, 2, 3])] = -3.0;

    assert!(e0.eq_epsilon(&e1, 1e-12));
  }

  #[test]
  fn multi_gramian_euclidean() {
    use common::gramian::Gramian;
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
