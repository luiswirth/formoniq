use geometry::{coord::Coord, metric::RiemannianMetric};
use index_algebra::{binomial, IndexSet};
use topology::Dim;

use crate::{ExteriorBasis, ExteriorRank, ExteriorTermExt};

pub type KVector = ExteriorElement;

pub type KForm = ExteriorElement;
pub type KFormCoordField<F> = ExteriorCoordField<F>;
pub type KFormCoeffs = ExteriorCoeffs;

pub type ExteriorCoeffs = na::DVector<f64>;

pub struct ExteriorCoordField<F>
where
  F: Fn(Coord) -> ExteriorCoeffs,
{
  coeff_fn: F,
  dim: Dim,
  rank: ExteriorRank,
}
impl<F> ExteriorCoordField<F>
where
  F: Fn(Coord) -> ExteriorCoeffs,
{
  pub fn dim(&self) -> Dim {
    self.dim
  }
  pub fn rank(&self) -> ExteriorRank {
    self.rank
  }
  pub fn at_point(&self, coord: Coord) -> ExteriorElement {
    let coeffs = (self.coeff_fn)(coord);
    ExteriorElement::new(coeffs, self.dim, self.rank)
  }
}

pub struct ExteriorElement {
  coeffs: na::DVector<f64>,
  dim: Dim,
  rank: ExteriorRank,
}

impl ExteriorElement {
  pub fn new(coeffs: na::DVector<f64>, dim: Dim, rank: ExteriorRank) -> Self {
    assert_eq!(coeffs.len(), binomial(dim, rank));
    Self { coeffs, dim, rank }
  }

  pub fn zero(dim: Dim, rank: ExteriorRank) -> Self {
    Self {
      coeffs: na::DVector::zeros(binomial(dim, rank)),
      dim,
      rank,
    }
  }

  pub fn one(dim: Dim) -> Self {
    Self {
      coeffs: na::DVector::from_element(1, 1.0),
      dim,
      rank: 0,
    }
  }

  // TODO: naming???
  pub fn from_1vector(vector: na::DVector<f64>) -> Self {
    let dim = vector.len();
    Self::new(vector, dim, 1)
  }

  pub fn into_1vector(self) -> na::DVector<f64> {
    assert!(self.rank == 1);
    self.coeffs
  }

  pub fn basis_iter(&self) -> impl Iterator<Item = (f64, ExteriorBasis)> + use<'_> {
    self
      .coeffs
      .iter()
      .copied()
      .enumerate()
      .map(move |(i, coeff)| {
        let basis = IndexSet::from_lex_rank(self.dim, self.rank, i).ext(self.dim);
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
          .index_set
          .clone()
          .union(other_basis.index_set.clone())
          .try_into_sorted_signed()
        {
          let sign = merged_basis.sign;
          let merged_basis = merged_basis.set.lex_rank(dim);
          new_coeffs[merged_basis] += sign.as_f64() * dbg!(self_coeff * other_coeff);
        }
      }
    }

    Self::new(new_coeffs, self.dim, new_rank)
  }

  pub fn wedge_big(factors: impl IntoIterator<Item = Self>) -> Self {
    let mut factors = factors.into_iter();
    let first = factors.next().unwrap();
    factors.fold(first, |acc, factor| acc.wedge(&factor))
  }

  pub fn hodge_star(&self, _metric: &RiemannianMetric) -> Self {
    todo!()
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
    assert_eq!(self.rank, other.rank);
    self.coeffs += other.coeffs;
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

impl KForm {
  pub fn on_kvector(&self, kvector: &KVector) -> f64 {
    assert!(self.dim == kvector.dim && self.rank == kvector.rank);
    self.coeffs.dot(&kvector.coeffs)
  }
}

impl std::ops::Index<ExteriorBasis> for ExteriorElement {
  type Output = f64;
  fn index(&self, index: ExteriorBasis) -> &Self::Output {
    assert!(index.rank() == self.rank);
    let index = index.index_set.lex_rank(self.dim);
    &self.coeffs[index]
  }
}
impl std::ops::Index<usize> for ExteriorElement {
  type Output = f64;
  fn index(&self, index: usize) -> &Self::Output {
    &self.coeffs[index]
  }
}

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

    let form_a = ExteriorElement::new(coeffs_a, dim, rank_1);
    let form_b = ExteriorElement::new(coeffs_b, dim, rank_2);

    // Perform the wedge product
    let result = form_a.wedge(&form_b);

    // Expected result: e1 ∧ e2 → a 2-form with coefficient 1 for the basis (1,2)
    let expected_coeffs = DVector::from_vec(vec![1.0, 0.0, 0.0]); // Basis: (1,2), (1,3), (2,3)
    let expected_form = ExteriorElement::new(expected_coeffs, dim, 2);

    assert_eq!(result.coeffs, expected_form.coeffs);
  }

  #[test]
  fn test_wedge_product_antisymmetry() {
    // Check antisymmetry: e1 ∧ e2 = -e2 ∧ e1
    let dim = 3;
    let rank = 1;

    let coeffs_a = DVector::from_vec(vec![1.0, 0.0, 0.0]); // e1
    let coeffs_b = DVector::from_vec(vec![0.0, 1.0, 0.0]); // e2

    let form_a = ExteriorElement::new(coeffs_a, dim, rank);
    let form_b = ExteriorElement::new(coeffs_b, dim, rank);

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

    let form_a = ExteriorElement::new(coeffs_a, dim, rank_1);
    let zero_form = ExteriorElement::new(coeffs_zero, dim, rank_2);

    let result = form_a.wedge(&zero_form);

    let expected_coeffs = DVector::zeros(3);
    let expected_form = ExteriorElement::new(expected_coeffs, dim, 2);

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

    let form_a = ExteriorElement::new(coeffs_a, dim, rank_1);
    let form_b = ExteriorElement::new(coeffs_b, dim, rank_2);

    let result = std::panic::catch_unwind(|| form_a.wedge(&form_b));
    assert!(result.is_err());
  }
}
