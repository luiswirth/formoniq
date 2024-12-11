use crate::{
  combo::{combinators::IndexSubsets, exterior::ExteriorTerm, variants::*},
  geometry::RiemannianMetric,
  Dim,
};

pub type ExteriorRank = usize;

#[derive(Debug, Clone)]
pub struct ScaledExteriorTerm<B: Base, O: Order> {
  coeff: f64,
  term: ExteriorTerm<B, O, Unsigned>,
}

impl<B: Base, O: Order> ScaledExteriorTerm<B, O> {
  pub fn from_raw(coeff: f64, term: impl Into<ExteriorTerm<B, O, Unsigned>>) -> Self {
    let term = term.into();
    Self { coeff, term }
  }

  pub fn coeff(&self) -> f64 {
    self.coeff
  }
  pub fn term(&self) -> &ExteriorTerm<B, O, Unsigned> {
    &self.term
  }

  pub fn rank(&self) -> ExteriorRank {
    self.term.len()
  }
  pub fn k(&self) -> ExteriorRank {
    self.rank()
  }

  pub fn forget_base(self) -> ScaledExteriorTerm<Unspecified, O> {
    let coeff = self.coeff;
    let term = self.term.forget_base();
    ScaledExteriorTerm { coeff, term }
  }

  pub fn into_canonical(self) -> ScaledExteriorTerm<B, Sorted> {
    let term = self.term.into_sorted();
    let coeff = self.coeff * term.sign().as_f64();
    let term = term.forget_sign();
    ScaledExteriorTerm { coeff, term }
  }

  pub fn assume_canonical(self) -> ScaledExteriorTerm<B, Sorted> {
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

/// With Local Base
impl<O: Order> ScaledExteriorTerm<Local, O> {
  pub fn dim(&self) -> Dim {
    self.term.base().len()
  }
  pub fn n(&self) -> Dim {
    self.dim()
  }
}

#[derive(Debug, Clone)]
pub struct ExteriorElement<B: Base, InnerO: Order, OuterO: Order> {
  terms: Vec<ScaledExteriorTerm<Unspecified, InnerO>>,
  base: B,
  term_order: OuterO,
}

impl<B: Base, OInner: Order> ExteriorElement<B, OInner, Ordered> {
  pub fn new(terms: Vec<ScaledExteriorTerm<B, OInner>>) -> Self {
    let base = terms[0].term().base().clone();
    let rank = terms[0].rank();
    assert!(terms.iter().all(|term| *term.term().base() == base));
    assert!(terms.iter().all(|term| term.rank() == rank));
    let terms = terms.into_iter().map(|t| t.forget_base()).collect();
    Self {
      terms,
      base,
      term_order: Ordered,
    }
  }
}

impl<B: Base, OInner: Order, OOuter: Order> ExteriorElement<B, OInner, OOuter> {
  pub fn rank(&self) -> ExteriorRank {
    self.terms[0].rank()
  }
  pub fn k(&self) -> ExteriorRank {
    self.rank()
  }

  pub fn term_order(&self) -> OOuter {
    self.term_order
  }

  pub fn into_canonical(self) -> ExteriorElement<B, Sorted, Sorted> {
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
      base: self.base,
      term_order: Sorted,
    }
  }

  pub fn assume_canonical(self) -> ExteriorElement<B, Sorted, Sorted> {
    let Self { terms, .. } = self;

    let terms: Vec<_> = terms
      .into_iter()
      .map(|term| term.assume_canonical())
      .collect();
    assert!(terms.is_sorted_by(|a, b| a.pure_lexicographical_cmp(b).is_lt()));

    ExteriorElement {
      terms,
      base: self.base,
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

/// With Local Base
impl<OInner: Order, OOuter: Order> ExteriorElement<Local, OInner, OOuter> {
  pub fn dim(&self) -> Dim {
    self.base.len()
  }
  pub fn n(&self) -> Dim {
    self.dim()
  }
}

impl<OuterO: Order> ExteriorElement<Local, Sorted, OuterO> {
  pub fn hodge_star(&self, metric: &RiemannianMetric) -> Self {
    let n = self.n();
    let k = self.k();
    let dual_k = n - k;

    let mut dual_terms = Vec::new();
    let mut dual_coeffs = Vec::new();
    for dual_index in IndexSubsets::canonical(n, dual_k) {
      let mut dual_coeff = 0.0;
      for self_term in self.terms.iter() {
        let self_coeff = self_term.coeff();
        let self_index = self_term.term();
        let mut dual_coeff_term = self_coeff;
        for iindex in 0..k {
          dual_coeff_term *= metric.covector_gramian()[(self_index[iindex], dual_index[iindex])]
        }
        dual_coeff += dual_coeff_term;
      }
      dual_coeff *= metric.det_sqrt();
      if dual_coeff != 0.0 {
        dual_coeffs.push(dual_coeff);
        dual_terms.push(dual_index);
      }
    }
    let terms = dual_terms
      .into_iter()
      .zip(dual_coeffs)
      .map(|(t, c)| ScaledExteriorTerm::from_raw(c, t).forget_base())
      .collect();

    Self {
      terms,
      base: self.base,
      term_order: self.term_order,
    }
  }
}

#[cfg(test)]
mod test {
  use crate::combo::IndexSet;

  #[test]
  fn canonical_conversion() {
    use super::*;

    let terms = ExteriorElement::new(vec![
      ScaledExteriorTerm::from_raw(1.0, vec![2, 0, 1]),
      ScaledExteriorTerm::from_raw(3.0, vec![1, 3, 2]),
      ScaledExteriorTerm::from_raw(-2.0, vec![0, 2, 1]),
      ScaledExteriorTerm::from_raw(3.0, vec![0, 1, 2]),
    ]);
    let canonical = terms.into_canonical();
    let expected = ExteriorElement::new(vec![
      ScaledExteriorTerm::from_raw(6.0, vec![0, 1, 2]),
      ScaledExteriorTerm::from_raw(-3.0, vec![1, 2, 3]),
    ])
    .assume_canonical();
    assert!(canonical.eq_epsilon(&expected, 10e-12));
  }

  #[test]
  fn hodge_star_euclidean() {
    use super::*;

    let dim = 3;
    let metric = RiemannianMetric::euclidean(dim);

    let term = ExteriorElement::new(vec![ScaledExteriorTerm::from_raw(
      1.0,
      IndexSet::new(vec![0]).assume_sorted().with_local_base(3),
    )])
    .into_canonical();
    let dual = term.hodge_star(&metric);
    dbg!(&dual);
    panic!();
  }
}
