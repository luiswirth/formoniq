use crate::Dim;

use index_algebra::{
  binomial,
  combinators::{IndexSubsets, IndexSupsets},
  variants::*,
  IndexSet, SignedIndexSet,
};

pub type Simplex<O> = IndexSet<O>;
pub type SignedSimplex<O> = SignedIndexSet<O>;

pub type SortedSimplex = Simplex<CanonicalOrder>;

pub trait SimplexExt<O: SetOrder> {
  fn dim(&self) -> Dim;
  fn subsimps(&self, sub_dim: Dim) -> IndexSubsets<O>;
}
impl<O: SetOrder> SimplexExt<O> for Simplex<O> {
  fn dim(&self) -> Dim {
    self.len() - 1
  }
  fn subsimps(&self, sub_dim: Dim) -> IndexSubsets<O> {
    self.subsets(sub_dim + 1)
  }
}

pub trait SortedSimplexExt {
  fn supsimps(&self, sub_dim: Dim, root: &Self) -> IndexSupsets;
}
impl SortedSimplexExt for SortedSimplex {
  fn supsimps(&self, sup_dim: Dim, root: &Self) -> IndexSupsets {
    self.supsets(sup_dim + 1, root)
  }
}

pub fn graded_subsimplicies(
  dim_facet: Dim,
) -> impl Iterator<Item = impl Iterator<Item = SortedSimplex>> {
  (0..=dim_facet).map(move |d| subsimplicies(dim_facet, d))
}
pub fn subsimplicies(dim_facet: Dim, dim_sub: Dim) -> impl Iterator<Item = SortedSimplex> {
  IndexSubsets::canonical(dim_facet + 1, dim_sub + 1)
}

pub fn nsubsimplicies(dim_facet: Dim, dim_sub: Dim) -> usize {
  binomial(dim_facet + 1, dim_sub + 1)
}
