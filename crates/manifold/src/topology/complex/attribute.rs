use super::{
  handle::{KSimplexIdx, SimplexHandle, SimplexIdx},
  Complex,
};
use crate::Dim;

use common::combo::Sign;

pub struct KSimplexCollection {
  kidxs: Vec<KSimplexIdx>,
  dim: Dim,
}
impl KSimplexCollection {
  pub fn new(idxs: Vec<KSimplexIdx>, dim: Dim) -> Self {
    Self { kidxs: idxs, dim }
  }
  pub fn kidxs(&self) -> &[KSimplexIdx] {
    &self.kidxs
  }
  pub fn len(&self) -> usize {
    self.kidxs.len()
  }
  pub fn is_empty(&self) -> bool {
    self.kidxs.is_empty()
  }
  pub fn kidx_iter(&self) -> impl ExactSizeIterator<Item = KSimplexIdx> + '_ {
    self.kidxs.iter().copied()
  }
  pub fn idx_iter(&self) -> impl ExactSizeIterator<Item = SimplexIdx> + '_ {
    self
      .kidx_iter()
      .map(|kidx| SimplexIdx::from((self.dim, kidx)))
  }
  pub fn handle_iter<'c>(
    &self,
    complex: &'c Complex,
  ) -> impl ExactSizeIterator<Item = SimplexHandle<'c>> + use<'c, '_> {
    self.idx_iter().map(|idx| SimplexHandle::new(complex, idx))
  }
}
impl FromIterator<SimplexIdx> for KSimplexCollection {
  fn from_iter<T: IntoIterator<Item = SimplexIdx>>(iter: T) -> Self {
    let mut dim = None;
    let kidxs = iter
      .into_iter()
      .map(|idx| {
        match dim {
          Some(dim) => assert_eq!(idx.dim, dim),
          None => dim = Some(idx.dim),
        }
        idx.kidx
      })
      .collect();
    Self::new(kidxs, dim.unwrap())
  }
}
impl<'a> FromIterator<SimplexHandle<'a>> for KSimplexCollection {
  fn from_iter<T: IntoIterator<Item = SimplexHandle<'a>>>(iter: T) -> Self {
    iter.into_iter().map(|handle| handle.idx()).collect()
  }
}

pub type SparseChain = SparseSkeletonAttributes<i32>;
pub type SparseSignChain = SparseSkeletonAttributes<Sign>;

pub struct SparseSkeletonAttributes<T> {
  data: Vec<T>,
  kidxs: Vec<KSimplexIdx>,
  dim: Dim,
}
impl<T> SparseSkeletonAttributes<T> {
  pub fn new(dim: Dim, idxs: Vec<KSimplexIdx>, data: Vec<T>) -> Self {
    Self {
      dim,
      kidxs: idxs,
      data,
    }
  }

  pub fn dim(&self) -> Dim {
    self.dim
  }

  pub fn data(&self) -> &[T] {
    &self.data
  }
  pub fn kidxs(&self) -> &[KSimplexIdx] {
    &self.kidxs
  }
  pub fn len(&self) -> usize {
    self.kidxs.len()
  }
  pub fn is_empty(&self) -> bool {
    self.len() == 0
  }
}
