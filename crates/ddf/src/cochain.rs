use common::linalg::nalgebra::{CsrMatrix, Vector};

use {
  exterior::ExteriorGrade,
  manifold::{
    topology::skeleton::Skeleton,
    topology::{
      complex::Complex,
      handle::{SimplexRef, SimplexIdx},
    },
  },
};

/// A $k$-cochain: one real coefficient per $k$-simplex of the skeleton.
///
/// An element of the cochain space $C^k$, hence a vector space over the
/// simplices of a fixed grade.
#[derive(Debug, Clone)]
pub struct Cochain {
  coeffs: Vector,
  grade: ExteriorGrade,
}
impl Cochain {
  pub fn new(grade: ExteriorGrade, coeffs: Vector) -> Self {
    Self { coeffs, grade }
  }
  pub fn constant(value: f64, skeleton: &Skeleton) -> Self {
    let ncoeffs = skeleton.len();
    Self::new(skeleton.dim(), Vector::from_element(ncoeffs, value))
  }
  pub fn zero(skeleton: &Skeleton) -> Self {
    Self::constant(0.0, skeleton)
  }
  pub fn from_function<F>(f: F, grade: ExteriorGrade, topology: &Complex) -> Self
  where
    F: FnMut(SimplexRef) -> f64,
  {
    let skeleton = topology.skeleton(grade);
    let coeffs = Vector::from_iterator(skeleton.len(), skeleton.handle_iter().map(f));
    Self::new(grade, coeffs)
  }

  pub fn grade(&self) -> ExteriorGrade {
    self.grade
  }
  pub fn coeffs(&self) -> &Vector {
    &self.coeffs
  }
  pub fn coeffs_mut(&mut self) -> &mut Vector {
    &mut self.coeffs
  }
  pub fn into_coeffs(self) -> Vector {
    self.coeffs
  }
  pub fn len(&self) -> usize {
    self.coeffs.len()
  }
  pub fn is_empty(&self) -> bool {
    self.coeffs.is_empty()
  }

  /// The discrete exterior derivative $dif: C^k -> C^(k+1)$: the coboundary
  /// operator applied to this cochain's coefficients.
  pub fn dif(&self, topology: &Complex) -> Self {
    let dif_operator = CsrMatrix::from(&topology.coboundary_operator(self.grade()));
    Cochain::new(self.grade() + 1, dif_operator * self.coeffs())
  }
}

impl std::ops::Index<SimplexIdx> for Cochain {
  type Output = f64;
  fn index(&self, idx: SimplexIdx) -> &Self::Output {
    assert_eq!(idx.dim(), self.grade());
    &self.coeffs[idx.kidx]
  }
}
impl std::ops::IndexMut<SimplexIdx> for Cochain {
  fn index_mut(&mut self, idx: SimplexIdx) -> &mut Self::Output {
    assert_eq!(idx.dim(), self.grade());
    &mut self.coeffs[idx.kidx]
  }
}

impl std::ops::Index<SimplexRef<'_>> for Cochain {
  type Output = f64;
  fn index(&self, handle: SimplexRef<'_>) -> &Self::Output {
    assert_eq!(handle.dim(), self.grade());
    &self.coeffs[handle.kidx()]
  }
}
impl std::ops::IndexMut<SimplexRef<'_>> for Cochain {
  fn index_mut(&mut self, idx: SimplexRef<'_>) -> &mut Self::Output {
    assert_eq!(idx.dim(), self.grade());
    &mut self.coeffs[idx.kidx()]
  }
}

impl std::ops::Index<usize> for Cochain {
  type Output = f64;
  fn index(&self, idx: usize) -> &Self::Output {
    &self.coeffs[idx]
  }
}

impl std::ops::Mul<f64> for Cochain {
  type Output = Cochain;
  fn mul(mut self, rhs: f64) -> Self::Output {
    self *= rhs;
    self
  }
}
impl std::ops::Mul<Cochain> for f64 {
  type Output = Cochain;
  fn mul(self, rhs: Cochain) -> Self::Output {
    rhs * self
  }
}
impl std::ops::MulAssign<f64> for Cochain {
  fn mul_assign(&mut self, rhs: f64) {
    self.coeffs *= rhs;
  }
}
impl std::ops::Neg for Cochain {
  type Output = Self;
  fn neg(self) -> Self::Output {
    Self::new(self.grade, -self.coeffs)
  }
}
impl std::ops::AddAssign for Cochain {
  fn add_assign(&mut self, rhs: Self) {
    assert_eq!(self.grade, rhs.grade);
    self.coeffs += rhs.coeffs;
  }
}
impl std::ops::Add for Cochain {
  type Output = Self;
  fn add(mut self, rhs: Self) -> Self::Output {
    self += rhs;
    self
  }
}
impl std::ops::SubAssign for Cochain {
  fn sub_assign(&mut self, rhs: Self) {
    assert_eq!(self.grade, rhs.grade);
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
