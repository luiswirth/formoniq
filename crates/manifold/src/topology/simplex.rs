use crate::Dim;

use multi_index::{
  binomial, combinators::IndexSubsets, sign::Sign, variants::*, MultiIndex, SignedIndexSet,
};

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct Simplex<O: IndexKind> {
  pub vertices: MultiIndex<O>,
}
impl<O: IndexKind> Simplex<O> {
  pub fn new(vertices: MultiIndex<O>) -> Self {
    Self { vertices }
  }

  pub fn nvertices(&self) -> usize {
    self.vertices.len()
  }
  pub fn dim(&self) -> Dim {
    self.nvertices() - 1
  }
  pub fn subsimps(&self, sub_dim: Dim) -> impl Iterator<Item = Self> {
    self.vertices.subsets(sub_dim + 1).map(Self::from)
  }

  pub fn boundary(&self) -> impl Iterator<Item = SignedSimplex<O>> {
    self.vertices.boundary().map(SignedSimplex::from)
  }

  pub fn with_sign(self, sign: Sign) -> SignedSimplex<O> {
    SignedSimplex::new(self.vertices, sign)
  }

  pub fn assume_sorted(self) -> Simplex<IncreasingSet> {
    Simplex::new(self.vertices.assume_sorted())
  }
  pub fn into_sorted(self) -> Simplex<IncreasingSet> {
    Simplex::new(self.vertices.into_sorted())
  }

  pub fn global_to_local_subsimp(&self, sub: &Self) -> Simplex<ArbitraryList> {
    Simplex::new(self.vertices.global_to_local_subset(&sub.vertices))
  }
}
impl<O: IndexKind, S: Into<MultiIndex<O>>> From<S> for Simplex<O> {
  fn from(vertices: S) -> Self {
    Self::new(vertices.into())
  }
}

pub type SortedSimplex = Simplex<IncreasingSet>;
impl SortedSimplex {
  pub fn standard(dim: Dim) -> Self {
    Self::new(MultiIndex::increasing(dim + 1))
  }

  pub fn supsimps(&self, sup_dim: Dim, root: &Self) -> impl Iterator<Item = Self> {
    self
      .vertices
      .supsets(sup_dim + 1, &root.vertices)
      .map(Self::from)
  }
  pub fn anti_boundary(
    &self,
    root: &SortedSimplex,
  ) -> impl Iterator<Item = SignedSimplex<IncreasingSet>> {
    self
      .vertices
      .anti_boundary(&root.vertices)
      .map(SignedSimplex::from)
  }

  pub fn is_subsimp_of(&self, other: &Self) -> bool {
    self.vertices.is_subset_of(&other.vertices)
  }
}

pub fn graded_subsimplicies(
  dim_cell: Dim,
) -> impl Iterator<Item = impl Iterator<Item = SortedSimplex>> {
  (0..=dim_cell).map(move |d| subsimplicies(dim_cell, d))
}
pub fn subsimplicies(dim_cell: Dim, dim_sub: Dim) -> impl Iterator<Item = SortedSimplex> {
  IndexSubsets::canonical(dim_cell + 1, dim_sub + 1).map(Simplex::new)
}

pub fn nsubsimplicies(dim_cell: Dim, dim_sub: Dim) -> usize {
  binomial(dim_cell + 1, dim_sub + 1)
}
pub fn nedges(dim_cell: Dim) -> usize {
  nsubsimplicies(dim_cell, 1)
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct SignedSimplex<O: IndexKind> {
  pub vertices: MultiIndex<O>,
  pub sign: Sign,
}
impl<O: IndexKind> SignedSimplex<O> {
  pub fn new(vertices: MultiIndex<O>, sign: Sign) -> Self {
    Self { vertices, sign }
  }

  pub fn into_simplex(self) -> Simplex<O> {
    Simplex::new(self.vertices)
  }
}
impl<O: IndexKind> From<SignedIndexSet<O>> for SignedSimplex<O> {
  fn from(signed_vertices: SignedIndexSet<O>) -> Self {
    Self::new(signed_vertices.set, signed_vertices.sign)
  }
}
