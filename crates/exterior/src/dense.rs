use crate::{
  variance::{self, VarianceMarker},
  ExteriorBasis, ExteriorRank, ExteriorTermExt, ScaledExteriorTerm,
};

use geometry::{coord::Coord, metric::RiemannianMetric};
use index_algebra::{binomial, variants::SetOrder, IndexSet};
use topology::Dim;

use std::marker::PhantomData;

/// An element of an exterior algebra.
pub struct ExteriorElement<V: VarianceMarker> {
  coeffs: na::DVector<f64>,
  dim: Dim,
  rank: ExteriorRank,
  variance: PhantomData<V>,
}

pub type KVector = ExteriorElement<variance::Contra>;
impl KVector {}

pub type KForm = ExteriorElement<variance::Co>;
impl KForm {
  /// Precompose k-form by some linear map.
  ///
  /// Needed for pullback/pushforward of differential k-form.
  pub fn precompose(&self, linear_map: &na::DMatrix<f64>) -> Self {
    self
      .basis_iter()
      .map(|(coeff, basis)| {
        coeff
          * KForm::wedge_big(
            basis
              .indices()
              .iter()
              .map(|i| KForm::from_rank1(linear_map.row(i).transpose())),
          )
          .unwrap_or(ExteriorElement::one(self.dim))
      })
      .sum()
  }

  pub fn evaluate(&self, kvector: &KVector) -> f64 {
    assert!(self.dim == kvector.dim && self.rank == kvector.rank);
    self.coeffs.dot(&kvector.coeffs)
  }

  pub fn hodge_star(&self, metric: &RiemannianMetric) -> Self {
    self
      .basis_iter()
      .map(|(coeff, basis)| (coeff * basis).hodge_star(metric))
      .fold(Self::one(self.dim), |acc, a| acc + a)
  }
}

impl<V: VarianceMarker> ExteriorElement<V> {
  pub fn new(coeffs: na::DVector<f64>, dim: Dim, rank: ExteriorRank) -> Self {
    assert_eq!(coeffs.len(), binomial(dim, rank));
    Self {
      coeffs,
      dim,
      rank,
      variance: PhantomData,
    }
  }
  pub fn zero(dim: Dim, rank: ExteriorRank) -> Self {
    Self {
      coeffs: na::DVector::zeros(binomial(dim, rank)),
      dim,
      rank,
      variance: PhantomData,
    }
  }
  pub fn one(dim: Dim) -> Self {
    Self {
      coeffs: na::DVector::from_element(1, 1.0),
      dim,
      rank: 0,
      variance: PhantomData,
    }
  }

  pub fn from_rank1(vector: na::DVector<f64>) -> Self {
    let dim = vector.len();
    Self::new(vector, dim, 1)
  }
  pub fn into_rank1(self) -> na::DVector<f64> {
    assert!(self.rank == 1);
    self.coeffs
  }

  pub fn dim(&self) -> Dim {
    self.dim
  }
  pub fn rank(&self) -> ExteriorRank {
    self.rank
  }
  pub fn coeffs(&self) -> &na::DVector<f64> {
    &self.coeffs
  }
  pub fn into_coeffs(self) -> na::DVector<f64> {
    self.coeffs
  }

  pub fn basis_iter(&self) -> impl Iterator<Item = (f64, ExteriorBasis<V>)> + use<'_, V> {
    let dim = self.dim;
    let rank = self.rank;
    self
      .coeffs
      .iter()
      .copied()
      .enumerate()
      .map(move |(i, coeff)| {
        let basis = IndexSet::from_lex_rank(dim, rank, i).ext(dim);
        (coeff, basis)
      })
  }

  pub fn basis_iter_mut(
    &mut self,
  ) -> impl Iterator<Item = (&mut f64, ExteriorBasis<V>)> + use<'_, V> {
    let dim = self.dim;
    let rank = self.rank;
    self.coeffs.iter_mut().enumerate().map(move |(i, coeff)| {
      let basis = IndexSet::from_lex_rank(dim, rank, i).ext(dim);
      (coeff, basis)
    })
  }

  pub fn wedge(&self, other: &Self) -> Self {
    assert_eq!(
      self.dim, other.dim,
      "Dimensions must match for wedge product"
    );
    let dim = self.dim;
    assert!(
      self.rank + other.rank <= self.dim,
      "Resultant rank exceeds the dimension of the space"
    );

    let new_rank = self.rank + other.rank;
    let new_basis_size = binomial(self.dim, new_rank);
    let mut new_coeffs = na::DVector::zeros(new_basis_size);

    for (self_coeff, self_basis) in self.basis_iter() {
      for (other_coeff, other_basis) in other.basis_iter() {
        if self_coeff == 0.0 || other_coeff == 0.0 {
          continue;
        }
        if self_basis == other_basis {
          continue;
        }

        if let Some(merged_basis) = self_basis
          .indices
          .clone()
          .union(other_basis.indices.clone())
          .try_into_sorted_signed()
        {
          let sign = merged_basis.sign;
          let merged_basis = merged_basis.set.lex_rank(dim);
          new_coeffs[merged_basis] += sign.as_f64() * self_coeff * other_coeff;
        }
      }
    }

    Self::new(new_coeffs, self.dim, new_rank)
  }

  pub fn wedge_big(factors: impl IntoIterator<Item = Self>) -> Option<Self> {
    let mut factors = factors.into_iter();
    let first = factors.next()?;
    let prod = factors.fold(first, |acc, factor| acc.wedge(&factor));
    Some(prod)
  }

  pub fn eq_epsilon(&self, other: &Self, eps: f64) -> bool {
    self.dim == other.dim
      && self.rank == other.rank
      && (&self.coeffs - &other.coeffs).norm_squared() <= eps.powi(2)
  }
}

impl<V: VarianceMarker> std::ops::Add<ExteriorElement<V>> for ExteriorElement<V> {
  type Output = Self;
  fn add(mut self, other: ExteriorElement<V>) -> Self::Output {
    self += other;
    self
  }
}
impl<V: VarianceMarker> std::ops::AddAssign<ExteriorElement<V>> for ExteriorElement<V> {
  fn add_assign(&mut self, other: ExteriorElement<V>) {
    assert_eq!(self.dim, other.dim);
    assert_eq!(self.rank, other.rank);
    self.coeffs += other.coeffs;
  }
}

impl<V: VarianceMarker> std::ops::Mul<f64> for ExteriorElement<V> {
  type Output = Self;
  fn mul(mut self, scalar: f64) -> Self::Output {
    self *= scalar;
    self
  }
}
impl<V: VarianceMarker> std::ops::MulAssign<f64> for ExteriorElement<V> {
  fn mul_assign(&mut self, scalar: f64) {
    self.coeffs *= scalar;
  }
}
impl<V: VarianceMarker> std::ops::Mul<ExteriorElement<V>> for f64 {
  type Output = ExteriorElement<V>;
  fn mul(self, rhs: ExteriorElement<V>) -> Self::Output {
    rhs * self
  }
}

impl<V: VarianceMarker, O: SetOrder> std::ops::AddAssign<ScaledExteriorTerm<V, O>>
  for ExteriorElement<V>
{
  fn add_assign(&mut self, term: ScaledExteriorTerm<V, O>) {
    let term = term.into_canonical();
    self[term.term] += term.coeff;
  }
}

impl<V: VarianceMarker, O: SetOrder> From<ScaledExteriorTerm<V, O>> for ExteriorElement<V> {
  fn from(term: ScaledExteriorTerm<V, O>) -> Self {
    let term = term.into_canonical();
    let mut element = Self::zero(term.dim(), term.rank());
    element[term.term] += term.coeff;
    element
  }
}

impl<V: VarianceMarker> std::ops::Index<ExteriorBasis<V>> for ExteriorElement<V> {
  type Output = f64;
  fn index(&self, index: ExteriorBasis<V>) -> &Self::Output {
    assert!(index.rank() == self.rank);
    let index = index.indices.lex_rank(self.dim);
    &self.coeffs[index]
  }
}
impl<V: VarianceMarker> std::ops::IndexMut<ExteriorBasis<V>> for ExteriorElement<V> {
  fn index_mut(&mut self, index: ExteriorBasis<V>) -> &mut Self::Output {
    let index = index.indices.lex_rank(self.dim);
    &mut self.coeffs[index]
  }
}
impl<V: VarianceMarker> std::ops::Index<usize> for ExteriorElement<V> {
  type Output = f64;
  fn index(&self, index: usize) -> &Self::Output {
    &self.coeffs[index]
  }
}

impl<V: VarianceMarker, O: SetOrder> std::iter::FromIterator<ScaledExteriorTerm<V, O>>
  for ExteriorElement<V>
{
  fn from_iter<T: IntoIterator<Item = ScaledExteriorTerm<V, O>>>(iter: T) -> Self {
    let mut iter = iter.into_iter();
    let first = iter.next().unwrap();
    let mut element = Self::from(first);
    iter.for_each(|term| element += term);
    element
  }
}
impl<V: VarianceMarker> std::iter::Sum for ExteriorElement<V> {
  fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
    let mut iter = iter.into_iter();
    let mut sum = iter.next().unwrap();
    for element in iter {
      sum += element;
    }
    sum
  }
}

pub struct Rank1Wedge<V: VarianceMarker> {
  factors: Vec<na::DVector<f64>>,
  variance: PhantomData<V>,
}

impl<V: VarianceMarker> Rank1Wedge<V> {
  pub fn new(factors: Vec<na::DVector<f64>>) -> Self {
    Self {
      factors,
      variance: PhantomData,
    }
  }
}

pub type VectorWedge = Rank1Wedge<variance::Contra>;
impl VectorWedge {}

pub type CovectorWedge = Rank1Wedge<variance::Co>;
impl CovectorWedge {
  pub fn evaluate(&self, vectors: &VectorWedge) -> f64 {
    let covectors = self;
    let mut mat = na::DMatrix::zeros(covectors.factors.len(), vectors.factors.len());
    for (i, covector) in self.factors.iter().enumerate() {
      for (j, vector) in vectors.factors.iter().enumerate() {
        mat[(i, j)] = covector.dot(vector);
      }
    }
    mat.determinant()
  }
}

pub struct ExteriorCoordField<V: VarianceMarker> {
  coeff_fn: Box<dyn Fn(Coord) -> na::DVector<f64>>,
  dim: Dim,
  rank: ExteriorRank,
  variance: PhantomData<V>,
}
impl<V: VarianceMarker> ExteriorCoordField<V> {
  pub fn dim(&self) -> Dim {
    self.dim
  }
  pub fn rank(&self) -> ExteriorRank {
    self.rank
  }
  pub fn at_point(&self, coord: Coord) -> ExteriorElement<V> {
    let coeffs = (self.coeff_fn)(coord);
    ExteriorElement::new(coeffs, self.dim, self.rank)
  }
}

pub type KVectorCoordField = ExteriorCoordField<variance::Contra>;
pub type KFormCoordField = ExteriorCoordField<variance::Co>;

#[cfg(test)]
mod tests {
  use super::*;
  use na::DVector;

  #[test]
  fn test_wedge_product_simple() {
    // Define two simple 1-forms in a 3D space
    let dim = 3;
    let rank_1 = 1;
    let rank_2 = 1;

    // Basis: e1, e2, e3
    let coeffs_a = DVector::from_vec(vec![1.0, 0.0, 0.0]); // Represents "e1"
    let coeffs_b = DVector::from_vec(vec![0.0, 1.0, 0.0]); // Represents "e2"

    let form_a = KForm::new(coeffs_a, dim, rank_1);
    let form_b = KForm::new(coeffs_b, dim, rank_2);

    // Perform the wedge product
    let result = form_a.wedge(&form_b);

    // Expected result: e1 ∧ e2 → a 2-form with coefficient 1 for the basis (1,2)
    let expected_coeffs = DVector::from_vec(vec![1.0, 0.0, 0.0]); // Basis: (1,2), (1,3), (2,3)
    let expected_form = KForm::new(expected_coeffs, dim, 2);

    assert_eq!(result.coeffs, expected_form.coeffs);
  }

  #[test]
  fn test_wedge_product_antisymmetry() {
    // Check antisymmetry: e1 ∧ e2 = -e2 ∧ e1
    let dim = 3;
    let rank = 1;

    let coeffs_a = DVector::from_vec(vec![1.0, 0.0, 0.0]); // e1
    let coeffs_b = DVector::from_vec(vec![0.0, 1.0, 0.0]); // e2

    let form_a = KForm::new(coeffs_a, dim, rank);
    let form_b = KForm::new(coeffs_b, dim, rank);

    let result_ab = form_a.wedge(&form_b);
    let result_ba = form_b.wedge(&form_a);

    // Antisymmetry: result_ab = -result_ba
    assert_eq!(result_ab.coeffs, -result_ba.coeffs);
  }

  #[test]
  fn test_wedge_product_with_zero_form() {
    // Wedge with a zero k-form should result in zero
    let dim = 3;
    let rank_1 = 1;
    let rank_2 = 1;

    let coeffs_a = DVector::from_vec(vec![1.0, 0.0, 0.0]); // e1
    let coeffs_zero = DVector::zeros(3); // Zero form

    let form_a = KForm::new(coeffs_a, dim, rank_1);
    let zero_form = KForm::new(coeffs_zero, dim, rank_2);

    let result = form_a.wedge(&zero_form);

    let expected_coeffs = DVector::zeros(3);
    let expected_form = KForm::new(expected_coeffs, dim, 2);

    assert_eq!(result.coeffs, expected_form.coeffs);
  }

  #[test]
  fn test_wedge_product_rank_exceeds_dim() {
    // Test that an assertion is triggered when rank exceeds the dimension
    let dim = 2;
    let rank_1 = 1;
    let rank_2 = 2;

    let coeffs_a = DVector::from_vec(vec![1.0, 0.0]);
    let coeffs_b = DVector::from_vec(vec![1.0]);

    let form_a = KForm::new(coeffs_a, dim, rank_1);
    let form_b = KForm::new(coeffs_b, dim, rank_2);

    let result = std::panic::catch_unwind(|| form_a.wedge(&form_b));
    assert!(result.is_err());
  }
}
