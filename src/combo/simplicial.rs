//! Combinatorial structures relevant for simplicial algebraic topology.

use super::{binomial, combinators::GradedIndexSubsets, variants::*, IndexSet};
use crate::Dim;

pub type Vertplex<B, O, S> = IndexSet<B, O, S>;

pub type SortedVertplex = Vertplex<Unspecified, Sorted, Unsigned>;
pub type OrderedVertplex = Vertplex<Unspecified, Ordered, Unsigned>;
pub type OrientedVertplex = Vertplex<Unspecified, Ordered, Signed>;

pub type RefVertplex = Vertplex<Local, Sorted, Unsigned>;

pub fn nsubsimplicies(dim: Dim, dim_sub: Dim) -> usize {
  let nverts = dim + 1;
  let nverts_sub = dim_sub + 1;
  binomial(nverts, nverts_sub)
}

pub fn subvertplexes(dim: Dim) -> Vec<Vec<RefVertplex>> {
  GradedIndexSubsets::canonical(dim + 1)
    .skip(1)
    .map(|subs| subs.collect())
    .collect()
}

pub trait SimplexExt {
  fn dim(&self) -> Dim;
  fn nvertices(&self) -> usize;
}
impl<B: Base, O: Order, S: Signedness> SimplexExt for Vertplex<B, O, S> {
  fn dim(&self) -> Dim {
    self.len() - 1
  }
  fn nvertices(&self) -> Dim {
    self.len()
  }
}
