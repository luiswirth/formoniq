use index_algebra::sign::Sign;

use super::{
  dim::RelDimTrait,
  handle::{KSimplexIdx, SimplexHandle, SimplexIdx},
  TopologyComplex,
};

pub struct KSimplexCollection<D: RelDimTrait> {
  kidxs: Vec<KSimplexIdx>,
  dim: D,
}
impl<D: RelDimTrait> KSimplexCollection<D> {
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
    self
      .kidx_iter()
      .map(|kidx| SimplexIdx::from((self.dim, kidx)))
  }
  pub fn handle_iter<'c>(
    &self,
    complex: &'c TopologyComplex,
  ) -> impl ExactSizeIterator<Item = SimplexHandle<'c, D>> + use<'c, '_, D> {
    self.idx_iter().map(|idx| SimplexHandle::new(complex, idx))
  }
}
impl<D: RelDimTrait> FromIterator<SimplexIdx<D>> for KSimplexCollection<D> {
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
    Self::new(kidxs, dim.unwrap_or(D::default()))
  }
}
impl<'a, D: RelDimTrait> FromIterator<SimplexHandle<'a, D>> for KSimplexCollection<D> {
  fn from_iter<T: IntoIterator<Item = SimplexHandle<'a, D>>>(iter: T) -> Self {
    iter.into_iter().map(|handle| handle.idx()).collect()
  }
}

pub type SparseChain<D> = SparseSkeletonAttributes<i32, D>;
pub type SparseSignChain<D> = SparseSkeletonAttributes<Sign, D>;

pub struct SparseSkeletonAttributes<T, D: RelDimTrait> {
  data: Vec<T>,
  kidxs: Vec<KSimplexIdx>,
  _dim: D,
}
impl<T, D: RelDimTrait> SparseSkeletonAttributes<T, D> {
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

pub struct Cochain<D: RelDimTrait> {
  pub coeffs: na::DVector<f64>,
  pub dim: D,
}
impl<D: RelDimTrait> Cochain<D> {
  pub fn new(dim: D, coeffs: na::DVector<f64>) -> Self {
    Self { dim, coeffs }
  }
  pub fn zero(dim: D, topology: &TopologyComplex) -> Self {
    let ncoeffs = topology.nsimplicies(dim);
    let coeffs = na::DVector::zeros(ncoeffs);
    Self::new(dim, coeffs)
  }

  pub fn coeffs(&self) -> &na::DVector<f64> {
    &self.coeffs
  }

  pub fn len(&self) -> usize {
    self.coeffs.len()
  }
  pub fn is_empty(&self) -> bool {
    self.coeffs().len() == 0
  }

  pub fn component_mul(&self, other: &Self) -> Self {
    assert_eq!(self.dim, other.dim);
    let coeffs = self.coeffs.component_mul(&other.coeffs);
    Self::new(self.dim, coeffs)
  }
}

impl<D: RelDimTrait> std::ops::Index<SimplexIdx<D>> for Cochain<D> {
  type Output = f64;
  fn index(&self, idx: SimplexIdx<D>) -> &Self::Output {
    &self.coeffs[idx.kidx]
  }
}

impl<D: RelDimTrait> std::ops::Index<SimplexHandle<'_, D>> for Cochain<D> {
  type Output = f64;
  fn index(&self, handle: SimplexHandle<'_, D>) -> &Self::Output {
    &self.coeffs[handle.kidx()]
  }
}

impl<D: RelDimTrait> std::ops::Index<usize> for Cochain<D> {
  type Output = f64;
  fn index(&self, idx: usize) -> &Self::Output {
    &self.coeffs[idx]
  }
}

impl<D: RelDimTrait> std::ops::SubAssign for Cochain<D> {
  fn sub_assign(&mut self, rhs: Self) {
    assert!(self.dim == rhs.dim);
    self.coeffs -= rhs.coeffs;
  }
}
impl<D: RelDimTrait> std::ops::Sub for Cochain<D> {
  type Output = Self;
  fn sub(mut self, rhs: Self) -> Self::Output {
    self -= rhs;
    self
  }
}
