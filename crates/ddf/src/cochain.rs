use common::linalg::nalgebra::{CsrMatrix, Vector};
use manifold::{
  geometry::{
    coord::{mesh::MeshCoords, quadrature::SimplexQuadRule},
    refsimp_vol,
  },
  topology::skeleton::Skeleton,
};

use crate::CoordSimplexExt;

use {
  exterior::{field::DifferentialMultiForm, ExteriorGrade},
  manifold::{
    geometry::coord::{simplex::SimplexCoords, CoordRef},
    topology::{
      complex::Complex,
      handle::{SimplexHandle, SimplexIdx},
    },
  },
};

#[derive(Debug, Clone)]
pub struct Cochain {
  pub coeffs: Vector,
  pub grade: ExteriorGrade,
}
impl Cochain {
  pub fn new(grade: ExteriorGrade, coeffs: Vector) -> Self {
    Self { grade, coeffs }
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
    F: FnMut(SimplexHandle) -> f64,
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
  pub fn len(&self) -> usize {
    self.coeffs.len()
  }
  pub fn is_empty(&self) -> bool {
    self.coeffs().len() == 0
  }

  pub fn dif(&self, topology: &Complex) -> Self {
    let dif_operator = CsrMatrix::from(&topology.coboundary_operator(self.grade()));
    Cochain::new(self.grade() + 1, dif_operator * self.coeffs())
  }

  /// Scale this cochain by a factor, modifying it in-place
  pub fn scale(&mut self, factor: f64) -> &mut Self {
    self.coeffs *= factor;
    self
  }

  /// Create a new cochain by scaling this one.
  pub fn scaled(&self, factor: f64) -> Self {
    Self::new(self.grade, &self.coeffs * factor)
  }

  pub fn component_mul(&self, other: &Self) -> Self {
    assert_eq!(self.grade, other.grade);
    let coeffs = self.coeffs.component_mul(&other.coeffs);
    Self::new(self.grade, coeffs)
  }
}

impl std::ops::Index<SimplexIdx> for Cochain {
  type Output = f64;
  fn index(&self, idx: SimplexIdx) -> &Self::Output {
    assert!(idx.dim() == self.grade());
    &self.coeffs[idx.kidx]
  }
}
impl std::ops::IndexMut<SimplexIdx> for Cochain {
  fn index_mut(&mut self, idx: SimplexIdx) -> &mut Self::Output {
    assert!(idx.dim() == self.grade());
    &mut self.coeffs[idx.kidx]
  }
}

impl std::ops::Index<SimplexHandle<'_>> for Cochain {
  type Output = f64;
  fn index(&self, handle: SimplexHandle<'_>) -> &Self::Output {
    assert!(handle.dim() == self.grade());
    &self.coeffs[handle.kidx()]
  }
}
impl std::ops::IndexMut<SimplexHandle<'_>> for Cochain {
  fn index_mut(&mut self, idx: SimplexHandle<'_>) -> &mut Self::Output {
    assert!(idx.dim() == self.grade());
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
  fn mul(self, rhs: f64) -> Self::Output {
    self.scaled(rhs)
  }
}
impl std::ops::MulAssign<f64> for Cochain {
  fn mul_assign(&mut self, rhs: f64) {
    self.scale(rhs);
  }
}
impl std::ops::SubAssign for Cochain {
  fn sub_assign(&mut self, rhs: Self) {
    assert!(self.grade == rhs.grade);
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

/// Discretize continuous coordinate-based differential k-form into
/// discrete k-cochain on CoordComplex via integration over k-simplex.
pub fn cochain_projection(
  form: &impl DifferentialMultiForm,
  topology: &Complex,
  coords: &MeshCoords,
  qr: Option<&SimplexQuadRule>,
) -> Cochain {
  let cochain = topology
    .skeleton(form.grade())
    .handle_iter()
    .map(|simp| SimplexCoords::from_simplex_and_coords(&simp, coords))
    .map(|simp| integrate_form_simplex(form, &simp, qr))
    .collect::<Vec<_>>()
    .into();
  Cochain::new(form.grade(), cochain)
}

/// Approximates the integral of a differential k-form over a k-simplex.
pub fn integrate_form_simplex(
  form: &impl DifferentialMultiForm,
  simplex: &SimplexCoords,
  qr: Option<&SimplexQuadRule>,
) -> f64 {
  let dim_intrinsic = simplex.dim_intrinsic();

  let multivector = simplex.spanning_multivector();
  let f = |coord: CoordRef| {
    form
      .at_point(simplex.local2global(coord).as_view())
      .apply_form_to_vector(&multivector)
  };
  if let Some(qr) = qr {
    qr.integrate_local(&f, refsimp_vol(dim_intrinsic))
  } else {
    let qr = &SimplexQuadRule::barycentric(dim_intrinsic);
    qr.integrate_local(&f, refsimp_vol(dim_intrinsic))
  }
}
