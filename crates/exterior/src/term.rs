use crate::{Dim, ExteriorElement, ExteriorGrade};

use common::{
  combo::{binomial, lex_rank, sort_signed, Sign},
  gramian::Gramian,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExteriorTerm {
  indices: Vec<usize>,
  dim: Dim,
}

impl ExteriorTerm {
  pub fn new(indices: Vec<usize>, dim: Dim) -> Self {
    Self { indices, dim }
  }
  pub fn top(dim: Dim) -> Self {
    Self::new((0..dim).collect(), dim)
  }

  pub fn indices(&self) -> &[usize] {
    &self.indices
  }
  pub fn grade(&self) -> ExteriorGrade {
    self.indices.len()
  }
  pub fn dim(&self) -> Dim {
    self.dim
  }

  pub fn is_basis(&self) -> bool {
    self.is_canonical()
  }
  pub fn is_canonical(&self) -> bool {
    let Some((sign, canonical)) = self.clone().canonicalized() else {
      return false;
    };
    sign == Sign::Pos && canonical == *self
  }
  pub fn canonicalized(mut self) -> Option<(Sign, Self)> {
    let sign = sort_signed(&mut self.indices);
    let len = self.indices.len();
    self.indices.dedup();
    if self.indices.len() != len {
      return None;
    }
    Some((sign, self))
  }

  pub fn wedge(mut self, mut other: Self) -> Self {
    self.indices.append(&mut other.indices);
    self
  }

  pub fn iter(&self) -> std::iter::Copied<std::slice::Iter<'_, usize>> {
    self.indices.iter().copied()
  }
}

impl std::ops::Index<usize> for ExteriorTerm {
  type Output = usize;
  fn index(&self, index: usize) -> &Self::Output {
    &self.indices[index]
  }
}

impl ExteriorTerm {
  pub fn from_lex_rank(dim: Dim, grade: ExteriorGrade, mut rank: usize) -> Self {
    let mut indices = Vec::with_capacity(grade);
    let mut start = 0;
    for i in 0..grade {
      let remaining = grade - i;
      for x in start..=(dim - remaining) {
        let c = binomial(dim - x - 1, remaining - 1);
        if rank < c {
          indices.push(x);
          start = x + 1;
          break;
        } else {
          rank -= c;
        }
      }
    }
    Self::new(indices, dim)
  }

  pub fn lex_rank(&self) -> usize {
    lex_rank(&self.indices, self.dim())
  }
}

impl std::ops::Mul<ExteriorTerm> for f64 {
  type Output = ExteriorElement;
  fn mul(self, term: ExteriorTerm) -> Self::Output {
    let coeff = self;
    coeff * ExteriorElement::from(term)
  }
}

pub fn exterior_bases(dim: Dim, grade: ExteriorGrade) -> impl Iterator<Item = ExteriorTerm> {
  itertools::Itertools::combinations(0..dim, grade)
    .map(move |indices| ExteriorTerm::new(indices, dim))
}

/// Construct Gramian on lexicographically ordered standard k-element standard
/// basis from Gramian on single elements.
///
/// The inner product on $Lambda^k$ is the exterior power of the inner product,
/// $inner(e_I, e_J)_(Lambda^k) = det [inner(e_i, e_j)]_(i in I, j in J)$.
pub fn multi_gramian(single_gramian: &Gramian, grade: ExteriorGrade) -> Gramian {
  Gramian::new_unchecked(crate::exterior_power(single_gramian.matrix(), grade))
}

#[cfg(test)]
mod test {
  use super::*;
  use approx::assert_relative_eq;
  use common::{combo::binomial, gramian::Gramian};

  #[test]
  fn canonical_conversion() {
    use super::ExteriorTerm as Ext;
    use super::*;

    let dim = 4;
    let mut e0 = ExteriorElement::zero(dim, 3);
    e0 += 1.0 * Ext::new(vec![2, 0, 1], dim);
    e0 += 3.0 * Ext::new(vec![1, 3, 2], dim);
    e0 += -2.0 * Ext::new(vec![0, 2, 1], dim);
    e0 += 3.0 * Ext::new(vec![0, 1, 2], dim);

    let mut e1 = ExteriorElement::zero(dim, 3);
    e1 += 6.0 * Ext::new(vec![0, 1, 2], dim);
    e1 += -3.0 * Ext::new(vec![1, 2, 3], dim);

    assert!(e0.eq_epsilon(&e1, 10e-12));
  }

  #[test]
  fn multi_gramian_euclidean() {
    for n in 0..=3 {
      let gramian = Gramian::standard(n);
      for k in 0..=n {
        let binomial = binomial(n, k);
        let expected_gram = Gramian::standard(binomial);
        let computed_gram = multi_gramian(&gramian, k);
        assert_relative_eq!(computed_gram.matrix(), expected_gram.matrix());
      }
    }
  }
}
