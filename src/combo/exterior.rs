//! Combinatorial structures relevant for exterior algebra.

use super::variants::*;
use super::IndexSet;

pub type ExteriorRank = usize;
pub type ExteriorTerm<B, O, S> = IndexSet<B, O, S>;

#[derive(Debug, Clone)]
pub struct ScaledExteriorTerm<O: Order> {
  coeff: f64,
  term: ExteriorTerm<Unspecified, O, Unsigned>,
}

impl<O: Order> ScaledExteriorTerm<O> {
  pub fn new(coeff: f64, term: impl Into<ExteriorTerm<Unspecified, O, Unsigned>>) -> Self {
    let term = term.into();
    Self { coeff, term }
  }

  pub fn rank(&self) -> ExteriorRank {
    self.term.len()
  }

  pub fn into_canonical(self) -> ScaledExteriorTerm<Sorted> {
    let term = self.term.into_sorted();
    let coeff = self.coeff * term.sign().as_f64();
    let term = term.forget_sign();
    ScaledExteriorTerm { coeff, term }
  }

  pub fn assume_canonical(self) -> ScaledExteriorTerm<Sorted> {
    ScaledExteriorTerm {
      coeff: self.coeff,
      term: self.term.assume_sorted(),
    }
  }

  pub fn pure_lexicographical_cmp(&self, other: &Self) -> std::cmp::Ordering {
    self.term.pure_lexicographical_cmp(&other.term)
  }

  pub fn eq_epsilon(&self, other: &Self, epsilon: f64) -> bool {
    self.term == other.term && (self.coeff - other.coeff).abs() < epsilon
  }
}

#[derive(Debug, Clone)]
pub struct ExteriorElement<OInner: Order, OOuter: Order> {
  terms: Vec<ScaledExteriorTerm<OInner>>,
  term_order: OOuter,
}

impl<OInner: Order> ExteriorElement<OInner, Ordered> {
  pub fn new(terms: Vec<ScaledExteriorTerm<OInner>>) -> Self {
    let rank = terms[0].rank();
    assert!(terms.iter().all(|term| term.rank() == rank));
    Self {
      terms,
      term_order: Ordered,
    }
  }
}

impl<OInner: Order, OOuter: Order> ExteriorElement<OInner, OOuter> {
  pub fn term_order(&self) -> OOuter {
    self.term_order
  }

  pub fn into_canonical(self) -> ExteriorElement<Sorted, Sorted> {
    let Self { terms, .. } = self;

    let mut terms: Vec<_> = terms
      .into_iter()
      .map(|term| term.into_canonical())
      .collect();
    terms.sort_unstable_by(ScaledExteriorTerm::pure_lexicographical_cmp);
    terms.dedup_by(|a, b| {
      if a.term == b.term {
        b.coeff += a.coeff;
        true
      } else {
        false
      }
    });

    ExteriorElement {
      terms,
      term_order: Sorted,
    }
  }

  pub fn assume_canonical(self) -> ExteriorElement<Sorted, Sorted> {
    let Self { terms, .. } = self;

    let terms: Vec<_> = terms
      .into_iter()
      .map(|term| term.assume_canonical())
      .collect();
    assert!(terms.is_sorted_by(|a, b| a.pure_lexicographical_cmp(b).is_lt()));

    ExteriorElement {
      terms,
      term_order: Sorted,
    }
  }

  pub fn eq_epsilon(&self, other: &Self, epsilon: f64) -> bool {
    self
      .terms
      .iter()
      .zip(other.terms.iter())
      .all(|(a, b)| a.eq_epsilon(b, epsilon))
  }
}

#[cfg(test)]
mod test {

  #[test]
  fn canonical_conversion() {
    use super::*;

    let terms = ExteriorElement::new(vec![
      ScaledExteriorTerm::new(1.0, vec![2, 0, 1]),
      ScaledExteriorTerm::new(3.0, vec![1, 3, 2]),
      ScaledExteriorTerm::new(-2.0, vec![0, 2, 1]),
      ScaledExteriorTerm::new(3.0, vec![0, 1, 2]),
    ]);
    let canonical = terms.into_canonical();
    let expected = ExteriorElement::new(vec![
      ScaledExteriorTerm::new(6.0, vec![0, 1, 2]),
      ScaledExteriorTerm::new(-3.0, vec![1, 2, 3]),
    ])
    .assume_canonical();
    assert!(canonical.eq_epsilon(&expected, 10e-12));
  }
}
