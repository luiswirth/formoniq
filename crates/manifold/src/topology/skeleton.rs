use super::{
  complex::handle::KSimplexIdx,
  simplex::{Simplex, SortedSimplex},
};
use crate::Dim;

use indexmap::IndexSet;

/// A container for simplicies of the same dimension.
#[derive(Default, Debug, Clone)]
pub struct Skeleton {
  simplicies: IndexSet<SortedSimplex>,
  nvertices: usize,
}
impl Skeleton {
  pub fn new(simplicies: Vec<SortedSimplex>) -> Self {
    Self::try_new(simplicies).unwrap()
  }
  pub fn try_new(simplicies: Vec<SortedSimplex>) -> Option<Self> {
    let dim = simplicies[0].dim();
    if !simplicies.iter().map(|simp| simp.dim()).all(|d| d == dim) {
      return None;
    }
    let nvertices = simplicies
      .iter()
      .map(|simp| simp.vertices.iter().max().unwrap())
      .max()
      .unwrap()
      + 1;

    let simplicies = IndexSet::from_iter(simplicies);
    Some(Self {
      simplicies,
      nvertices,
    })
  }
  pub fn standard(dim: Dim) -> Skeleton {
    Self::new(vec![Simplex::standard(dim)])
  }

  #[must_use]
  pub fn len(&self) -> usize {
    self.simplicies.len()
  }
  #[must_use]
  pub fn is_empty(&self) -> bool {
    self.len() == 0
  }
  #[must_use]
  pub fn dim(&self) -> Dim {
    self.simplicies[0].dim()
  }
  pub fn nvertices(&self) -> usize {
    self.nvertices
  }
  #[must_use]
  pub fn simplicies(&self) -> &IndexSet<SortedSimplex> {
    &self.simplicies
  }
  #[must_use]
  pub fn iter(&self) -> indexmap::set::Iter<'_, SortedSimplex> {
    self.simplicies.iter()
  }
  pub fn insert(&mut self, simp: SortedSimplex) -> (KSimplexIdx, bool) {
    self.simplicies.insert_full(simp)
  }
  pub fn into_index_set(self) -> IndexSet<SortedSimplex> {
    self.simplicies
  }

  pub fn simplex_by_kidx(&self, idx: KSimplexIdx) -> &SortedSimplex {
    self.simplicies.get_index(idx).unwrap()
  }
  pub fn kidx_by_simplex(&self, simp: &SortedSimplex) -> KSimplexIdx {
    self.simplicies.get_index_of(simp).unwrap()
  }
}
