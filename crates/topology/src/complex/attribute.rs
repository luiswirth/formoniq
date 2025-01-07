use index_algebra::sign::Sign;

use super::{
  dim::DimInfoProvider,
  handle::{KSimplexIdx, SimplexHandle, SimplexIdx},
  ManifoldComplex,
};

pub struct KSimplexCollection<D: DimInfoProvider> {
  kidxs: Vec<KSimplexIdx>,
  dim: D,
}
impl<D: DimInfoProvider> KSimplexCollection<D> {
  pub fn new(idxs: Vec<KSimplexIdx>, dim: D) -> Self {
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
  pub fn idx_iter(&self) -> impl ExactSizeIterator<Item = SimplexIdx<D>> + '_ {
    self.kidx_iter().map(|kidx| SimplexIdx::new(self.dim, kidx))
  }
  pub fn handle_iter<'c>(
    &self,
    complex: &'c ManifoldComplex,
  ) -> impl ExactSizeIterator<Item = SimplexHandle<'c, D>> + use<'c, '_, D> {
    self.idx_iter().map(|idx| SimplexHandle::new(complex, idx))
  }
}
impl<D: DimInfoProvider> FromIterator<SimplexIdx<D>> for KSimplexCollection<D> {
  fn from_iter<T: IntoIterator<Item = SimplexIdx<D>>>(iter: T) -> Self {
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
impl<'a, D: DimInfoProvider> FromIterator<SimplexHandle<'a, D>> for KSimplexCollection<D> {
  fn from_iter<T: IntoIterator<Item = SimplexHandle<'a, D>>>(iter: T) -> Self {
    iter.into_iter().map(|handle| handle.idx()).collect()
  }
}

pub type SparseChain<D> = SparseSkeletonAttributes<i32, D>;
pub type SparseSignChain<D> = SparseSkeletonAttributes<Sign, D>;

pub struct SparseSkeletonAttributes<T, D: DimInfoProvider> {
  data: Vec<T>,
  kidxs: Vec<KSimplexIdx>,
  _dim: D,
}
impl<T, D: DimInfoProvider> SparseSkeletonAttributes<T, D> {
  pub fn new(dim: D, idxs: Vec<KSimplexIdx>, data: Vec<T>) -> Self {
    Self {
      _dim: dim,
      kidxs: idxs,
      data,
    }
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

pub struct Cochain<D: DimInfoProvider> {
  pub coeffs: na::DVector<f64>,
  pub dim: D,
}
impl<D: DimInfoProvider> Cochain<D> {
  pub fn new(dim: D, coeffs: na::DVector<f64>) -> Self {
    Self { dim, coeffs }
  }
}
