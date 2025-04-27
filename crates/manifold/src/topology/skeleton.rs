use super::{handle::KSimplexIdx, simplex::Simplex};
use crate::Dim;

use indexmap::IndexSet;

/// A container for sorted simplices of the same dimension.
#[derive(Default, Debug, Clone)]
pub struct Skeleton {
  /// Every simplex is sorted.
  simplices: IndexSet<Simplex>,
  nvertices: usize,
}
impl Skeleton {
  /// Every simplex must be sorted.
  pub fn new(simplices: Vec<Simplex>) -> Self {
    assert!(!simplices.is_empty(), "Skeleton must not be empty");
    let dim = simplices[0].dim();
    assert!(
      simplices.iter().map(|simp| simp.dim()).all(|d| d == dim),
      "Skeleton simplices must have same dimension."
    );
    assert!(
      simplices.iter().all(|simp| simp.is_sorted()),
      "Skeleton simplices must be sorted."
    );
    let nvertices = if dim == 0 {
      assert!(simplices.iter().enumerate().all(|(i, simp)| simp[0] == i));
      simplices.len()
    } else {
      simplices
        .iter()
        .map(|simp| simp.iter().max().expect("Simplex is not empty."))
        .max()
        .expect("Simplices is not empty.")
        + 1
    };

    let simplices = IndexSet::from_iter(simplices);
    Self {
      simplices,
      nvertices,
    }
  }
  pub fn standard(dim: Dim) -> Skeleton {
    Self::new(vec![Simplex::standard(dim)])
  }

  #[must_use]
  pub fn len(&self) -> usize {
    self.simplices.len()
  }
  #[must_use]
  pub fn is_empty(&self) -> bool {
    self.len() == 0
  }
  #[must_use]
  pub fn dim(&self) -> Dim {
    self.simplices[0].dim()
  }
  pub fn nvertices(&self) -> usize {
    self.nvertices
  }
  #[must_use]
  pub fn simplices(&self) -> &IndexSet<Simplex> {
    &self.simplices
  }
  #[must_use]
  pub fn iter(&self) -> indexmap::set::Iter<'_, Simplex> {
    self.simplices.iter()
  }

  pub fn insert(&mut self, simp: Simplex) -> (KSimplexIdx, bool) {
    assert!(simp.is_sorted());
    self.simplices.insert_full(simp)
  }
  pub fn into_index_set(self) -> IndexSet<Simplex> {
    self.simplices
  }

  pub fn simplex_by_kidx(&self, idx: KSimplexIdx) -> &Simplex {
    self.simplices.get_index(idx).unwrap()
  }
  pub fn kidx_by_simplex(&self, simp: &Simplex) -> KSimplexIdx {
    self.simplices.get_index_of(simp).unwrap()
  }
}

impl IntoIterator for Skeleton {
  type Item = Simplex;
  type IntoIter = indexmap::set::IntoIter<Self::Item>;
  fn into_iter(self) -> Self::IntoIter {
    self.simplices.into_iter()
  }
}
