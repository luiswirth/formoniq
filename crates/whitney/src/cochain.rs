use manifold::topology::skeleton::Skeleton;

use crate::CoordSimplexExt;

use {
  exterior::{field::DifferentialMultiForm, Dim},
  manifold::{
    geometry::coord::{
      local::SimplexCoords, quadrature::barycentric_quadrature, CoordRef, MeshVertexCoords,
    },
    topology::complex::{
      handle::{SimplexHandle, SimplexIdx},
      Complex,
    },
  },
};

pub struct Cochain {
  pub coeffs: na::DVector<f64>,
  pub dim: Dim,
}
impl Cochain {
  pub fn new(dim: Dim, coeffs: na::DVector<f64>) -> Self {
    Self { dim, coeffs }
  }
  pub fn constant(value: f64, skeleton: &Skeleton) -> Self {
    let ncoeffs = skeleton.len();
    Self::new(skeleton.dim(), na::DVector::from_element(ncoeffs, value))
  }
  pub fn zero(skeleton: &Skeleton) -> Self {
    Self::constant(0.0, skeleton)
  }
  pub fn from_function<F>(f: F, dim: Dim, topology: &Complex) -> Self
  where
    F: FnMut(SimplexHandle) -> f64,
  {
    let skeleton = topology.skeleton(dim);
    let coeffs = na::DVector::from_iterator(skeleton.len(), skeleton.handle_iter().map(f));
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

  /// Scale this cochain by a factor, modifying it in-place
  pub fn scale(&mut self, factor: f64) -> &mut Self {
    self.coeffs *= factor;
    self
  }

  /// Create a new cochain by scaling this one.
  pub fn scaled(&self, factor: f64) -> Self {
    Self::new(self.dim, &self.coeffs * factor)
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
impl std::ops::IndexMut<SimplexIdx> for Cochain {
  fn index_mut(&mut self, idx: SimplexIdx) -> &mut Self::Output {
    assert!(idx.dim() == self.dim());
    &mut self.coeffs[idx.kidx]
  }
}

impl std::ops::Index<SimplexHandle<'_>> for Cochain {
  type Output = f64;
  fn index(&self, handle: SimplexHandle<'_>) -> &Self::Output {
    assert!(handle.dim() == self.dim());
    &self.coeffs[handle.kidx()]
  }
}
impl std::ops::IndexMut<SimplexHandle<'_>> for Cochain {
  fn index_mut(&mut self, idx: SimplexHandle<'_>) -> &mut Self::Output {
    assert!(idx.dim() == self.dim());
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

/// Discretize continuous coordinate-based differential k-form into
/// discrete k-cochain on CoordComplex via de Rham map (integration over k-simplex).
pub fn de_rham_map(
  form: &impl DifferentialMultiForm,
  topology: &Complex,
  coords: &MeshVertexCoords,
) -> Cochain {
  let cochain = topology
    .skeleton(form.grade())
    .handle_iter()
    .map(|simp| SimplexCoords::from_simplex_and_coords(simp.raw(), coords))
    .map(|simp| de_rahm_map_local(form, &simp))
    .collect::<Vec<_>>()
    .into();
  Cochain::new(form.grade(), cochain)
}

/// Approximates the integral of a differential k-form over a k-simplex,
/// by means of barycentric quadrature.
pub fn de_rahm_map_local(
  differential_form: &impl DifferentialMultiForm,
  simplex: &SimplexCoords,
) -> f64 {
  let multivector = simplex.spanning_multivector();
  let f = |coord: CoordRef| {
    differential_form
      .at_point(simplex.local2global(coord).as_view())
      .apply_form_on_multivector(&multivector)
  };
  let std_simp = SimplexCoords::standard(simplex.dim_intrinsic());
  barycentric_quadrature(&f, &std_simp)
}
