use exterior::Dim;
use manifold::topology::complex::{
  handle::{SimplexHandle, SimplexIdx},
  Complex,
};

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
