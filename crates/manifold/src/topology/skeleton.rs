use super::{complex::handle::KSimplexIdx, simplex::Simplex};
use crate::Dim;

use indexmap::IndexSet;

/// A container for sorted simplicies of the same dimension.
#[derive(Default, Debug, Clone)]
pub struct Skeleton {
  /// Every simplex is sorted.
  simplicies: IndexSet<Simplex>,
  nvertices: usize,
}
impl Skeleton {
  /// Every simplex must be sorted.
  pub fn new(simplicies: Vec<Simplex>) -> Self {
    assert!(!simplicies.is_empty(), "Skeleton must not be empty");
    let dim = simplicies[0].dim();
    assert!(
      simplicies.iter().map(|simp| simp.dim()).all(|d| d == dim),
      "Skeleton simplicies must have same dimension."
    );
    assert!(
      simplicies.iter().all(|simp| simp.is_sorted()),
      "Skeleton simplicies must be sorted."
    );
    let nvertices = if dim == 0 {
      assert!(simplicies.iter().enumerate().all(|(i, simp)| simp[0] == i));
      simplicies.len()
    } else {
      simplicies
        .iter()
        .map(|simp| simp.iter().max().expect("Simplex is not empty."))
        .max()
        .expect("Simplicies is not empty.")
        + 1
    };

    let simplicies = IndexSet::from_iter(simplicies);
    Self {
      simplicies,
      nvertices,
    }
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
  pub fn simplicies(&self) -> &IndexSet<Simplex> {
    &self.simplicies
  }
  #[must_use]
  pub fn iter(&self) -> indexmap::set::Iter<'_, Simplex> {
    self.simplicies.iter()
  }

  pub fn insert(&mut self, simp: Simplex) -> (KSimplexIdx, bool) {
    assert!(simp.is_sorted());
    self.simplicies.insert_full(simp)
  }
  pub fn into_index_set(self) -> IndexSet<Simplex> {
    self.simplicies
  }

  pub fn simplex_by_kidx(&self, idx: KSimplexIdx) -> &Simplex {
    self.simplicies.get_index(idx).unwrap()
  }
  pub fn kidx_by_simplex(&self, simp: &Simplex) -> KSimplexIdx {
    self.simplicies.get_index_of(simp).unwrap()
  }
}

impl IntoIterator for Skeleton {
  type Item = Simplex;
  type IntoIter = indexmap::set::IntoIter<Self::Item>;
  fn into_iter(self) -> Self::IntoIter {
    self.simplicies.into_iter()
  }
}
