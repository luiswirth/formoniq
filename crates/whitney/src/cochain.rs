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
    .map(|simp| SimplexCoords::from_simplex_and_coords(simp.simplex_set(), coords))
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
      .at_point(simplex.local_to_global_coord(coord).as_view())
      .on_multivector(&multivector)
  };
  let std_simp = SimplexCoords::standard(simplex.dim_intrinsic());
  barycentric_quadrature(&f, &std_simp)
}
