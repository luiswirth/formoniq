use crate::{
  simplex::{SimplexExt, SortedSimplex},
  Dim,
};

/// A container for simplicies of the same dimension.
#[derive(Debug, Clone)]
pub struct ManifoldSkeleton {
  simplicies: Vec<SortedSimplex>,
}
impl ManifoldSkeleton {
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
