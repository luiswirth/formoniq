//! Combinatorial structures relevant for simplicial algebraic topology.

use super::{binomial, combinators::GradedIndexSubsets, variants::*, IndexSet};
use crate::Dim;

pub type Vertplex<B, O, S> = IndexSet<B, O, S>;

pub type SortedVertplex<B> = Vertplex<B, Sorted, Unsigned>;
pub type OrderedVertplex<B> = Vertplex<B, Ordered, Unsigned>;
pub type OrientedVertplex<B> = Vertplex<B, Ordered, Signed>;

pub type RefVertplex = Vertplex<Local, Sorted, Unsigned>;

pub type MeshVertplex = Vertplex<Global, Sorted, Unsigned>;
pub type MeshCellVertplex = Vertplex<Global, Ordered, Signed>;

pub type MeshVertplexInner = Vertplex<Unspecified, Sorted, Unsigned>;
pub type MeshCellVertplexInner = Vertplex<Unspecified, Ordered, Signed>;

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
    self.k() - 1
  }
  fn nvertices(&self) -> Dim {
    self.k()
  }
}
