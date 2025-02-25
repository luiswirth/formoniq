use multi_index::sign::Sign;

use crate::Dim;

use super::{
  handle::{KSimplexIdx, SimplexHandle, SimplexIdx},
  Complex,
};

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

pub struct Cochain {
  pub coeffs: na::DVector<f64>,
  pub dim: Dim,
}
impl Cochain {
  pub fn new(dim: Dim, coeffs: na::DVector<f64>) -> Self {
    Self { dim, coeffs }
  }
  pub fn zero(dim: Dim, topology: &Complex) -> Self {
    let ncoeffs = topology.nsimplicies(dim);
    let coeffs = na::DVector::zeros(ncoeffs);
    Self::new(dim, coeffs)
  }
  pub fn dim(&self) -> Dim {
    self.dim
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

impl std::ops::Index<SimplexIdx> for Cochain {
  type Output = f64;
  fn index(&self, idx: SimplexIdx) -> &Self::Output {
    assert!(idx.dim() == self.dim());
    &self.coeffs[idx.kidx]
  }
}

impl std::ops::Index<SimplexHandle<'_>> for Cochain {
  type Output = f64;
  fn index(&self, handle: SimplexHandle<'_>) -> &Self::Output {
    assert!(handle.dim() == self.dim());
    &self.coeffs[handle.kidx()]
  }
}

impl std::ops::Index<usize> for Cochain {
  type Output = f64;
  fn index(&self, idx: usize) -> &Self::Output {
    &self.coeffs[idx]
  }
}

impl std::ops::SubAssign for Cochain {
  fn sub_assign(&mut self, rhs: Self) {
    assert!(self.dim == rhs.dim);
    self.coeffs -= rhs.coeffs;
  }
}
impl std::ops::Sub for Cochain {
  type Output = Self;
  fn sub(mut self, rhs: Self) -> Self::Output {
    self -= rhs;
    self
  }
}
