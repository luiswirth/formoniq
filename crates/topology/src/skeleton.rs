use crate::{simplex::SortedSimplex, Dim};

/// A container for simplicies of the same dimension.
#[derive(Debug, Clone)]
pub struct TopologySkeleton {
  simplicies: Vec<SortedSimplex>,
}
impl TopologySkeleton {
  pub fn new(simplicies: Vec<SortedSimplex>) -> Self {
    Self::try_new(simplicies).unwrap()
  }
  pub fn try_new(simplicies: Vec<SortedSimplex>) -> Option<Self> {
    let dim = simplicies[0].dim();

    if !simplicies.iter().map(|f| f.dim()).all(|d| d == dim) {
      return None;
    }

    Some(Self { simplicies })
  }

  #[must_use]
  pub fn len(&self) -> usize {
    self.simplicies.len()
  }
  #[must_use]
  pub fn is_empty(&self) -> bool {
    self.len() == 0
  }

  pub fn dim(&self) -> Dim {
    self.simplicies[0].dim()
  }
  pub fn simplicies(&self) -> &[SortedSimplex] {
    &self.simplicies
  }
  pub fn simplex_iter(&self) -> std::slice::Iter<'_, SortedSimplex> {
    self.simplicies.iter()
  }
  pub fn into_simplicies(self) -> Vec<SortedSimplex> {
    self.simplicies
  }
  pub fn into_simplex_iter(self) -> std::vec::IntoIter<SortedSimplex> {
    self.simplicies.into_iter()
  }
}
