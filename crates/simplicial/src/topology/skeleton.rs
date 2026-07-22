use super::{handle::KSimplexIdx, simplex::Simplex};
use crate::Dim;

use indexmap::IndexSet;

#[cfg(feature = "serde")]
use std::{io, path::Path};

/// The simplices of one fixed dimension, canonically colexicographically
/// ordered and deduplicated. Position in that order is the [`KSimplexIdx`],
/// the index every handle and boundary operator refers to.
#[derive(Default, Debug, Clone)]
pub struct Skeleton {
  simplices: IndexSet<Simplex>,
  nvertices: usize,
}

/// Serializes as the plain simplex list; deserialization runs it back through
/// [`Skeleton::new`], so the derived fields (`nvertices`, canonical order) are
/// always recomputed from the simplices, never trusted from the file.
#[cfg(feature = "serde")]
impl serde::Serialize for Skeleton {
  fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
    self
      .simplices
      .iter()
      .cloned()
      .collect::<Vec<_>>()
      .serialize(serializer)
  }
}
#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for Skeleton {
  fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
    Ok(Self::new(Vec::<Simplex>::deserialize(deserializer)?))
  }
}
impl Skeleton {
  pub fn new(mut simplices: Vec<Simplex>) -> Self {
    assert!(!simplices.is_empty(), "Skeleton must not be empty");
    let dim = simplices[0].dim();
    assert!(
      simplices
        .iter()
        .map(super::simplex::Simplex::dim)
        .all(|d| d == dim),
      "Skeleton simplices must have same dimension."
    );
    // Canonical ordering: colexicographic by vertex set, deduplicated.
    simplices.sort_unstable();
    simplices.dedup();
    let nvertices = if dim == 0 {
      assert!(
        simplices.iter().enumerate().all(|(i, simp)| simp[0] == i),
        "0-simplices must be the contiguous vertices 0..n."
      );
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

  pub fn standard(dim: impl Into<Dim>) -> Skeleton {
    let dim = dim.into();
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

  #[cfg(feature = "serde")]
  pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
    crate::io::cbor::save_cbor(self, path)
  }
  #[cfg(feature = "serde")]
  pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
    crate::io::cbor::load_cbor(path)
  }
}

impl IntoIterator for Skeleton {
  type Item = Simplex;
  type IntoIter = indexmap::set::IntoIter<Self::Item>;
  fn into_iter(self) -> Self::IntoIter {
    self.simplices.into_iter()
  }
}
