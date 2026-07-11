//! The de Rham map $R: L^2 Lambda^k -> C^k$.
//!
//! Discretization of differential forms into cochains by integration over
//! the simplices of the mesh. Together with the Whitney interpolation
//! $W: C^k -> L^2 Lambda^k$ (see [`crate::whitney`]) it forms the pair of
//! cochain maps at the heart of FEEC. The governing laws are executable:
//!
//! - $R compose W = id$: Whitney's theorem
//!   (test `whitney_basis_property` in [`crate`]).
//! - $R compose dif = dif compose R$: Stokes' theorem
//!   (test `derham_map_is_cochain_map`, below).
//! - $dif compose W = W compose dif$: Whitney forms are a subcomplex
//!   (test `whitney_interpolation_is_cochain_map` in [`crate::whitney`]).

use crate::{cochain::Cochain, CoordSimplexExt};

use {
  exterior::field::DifferentialMultiForm,
  manifold::{
    geometry::{
      coord::{mesh::MeshCoords, quadrature::SimplexQuadRule, simplex::SimplexCoords, CoordRef},
      refsimp_vol,
    },
    topology::complex::Complex,
  },
};

/// The de Rham map: discretize a continuous differential k-form into a
/// k-cochain by integrating it over each k-simplex of the mesh.
pub fn derham_map(
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

#[cfg(test)]
mod test {
  use super::*;

  use {
    common::linalg::nalgebra::Vector,
    exterior::{
      field::{DiffFormClosure, ExteriorField},
      ExteriorElement,
    },
    manifold::gen::cartesian::CartesianMeshInfo,
  };


  use approx::assert_relative_eq;

  /// $R compose dif = dif compose R$: the de Rham map is a cochain map.
  ///
  /// This is Stokes' theorem: integrating $dif omega$ over a simplex equals
  /// summing $omega$ over its boundary. Forms with affine coefficients make
  /// the barycentric quadrature exact, so both sides agree up to roundoff.
  #[test]
  fn derham_map_is_cochain_map() {
    // (omega, dif omega) pairs of polynomial differential forms.
    let cases: Vec<(DiffFormClosure, DiffFormClosure)> = vec![
      // 1d: omega = x^2, dif omega = 2x dx
      // (0-forms are evaluated, not integrated, so degree 2 stays exact)
      (
        DiffFormClosure::scalar(|p| p[0] * p[0], 1),
        DiffFormClosure::one_form(|p| Vector::from_element(1, 2.0 * p[0]), 1),
      ),
      // 2d: omega = x y, dif omega = y dx + x dy
      (
        DiffFormClosure::scalar(|p| p[0] * p[1], 2),
        DiffFormClosure::one_form(|p| na::dvector![p[1], p[0]], 2),
      ),
      // 2d: omega = y dx, dif omega = -dx wedge dy
      (
        DiffFormClosure::one_form(|p| na::dvector![p[1], 0.0], 2),
        DiffFormClosure::new(
          Box::new(|_| ExteriorElement::new(na::dvector![-1.0], 2, 2)),
          2,
          2,
        ),
      ),
      // 3d: omega = z dy, dif omega = -dy wedge dz
      (
        DiffFormClosure::one_form(|p| na::dvector![0.0, p[2], 0.0], 3),
        DiffFormClosure::new(
          Box::new(|_| ExteriorElement::new(na::dvector![0.0, 0.0, -1.0], 3, 2)),
          3,
          2,
        ),
      ),
      // 3d: omega = x dy wedge dz, dif omega = dx wedge dy wedge dz
      (
        DiffFormClosure::new(
          Box::new(|p| ExteriorElement::new(na::dvector![0.0, 0.0, p[0]], 3, 2)),
          3,
          2,
        ),
        DiffFormClosure::new(
          Box::new(|_| ExteriorElement::new(na::dvector![1.0], 3, 3)),
          3,
          3,
        ),
      ),
    ];

    for (form, dif_form) in cases {
      let dim = form.dim_intrinsic();
      let (topology, coords) = CartesianMeshInfo::new_unit(dim, 2).compute_coord_complex();

      let dif_of_projected = derham_map(&form, &topology, &coords, None).dif(&topology);
      let projected_dif = derham_map(&dif_form, &topology, &coords, None);

      assert_eq!(dif_of_projected.grade, projected_dif.grade);
      assert_relative_eq!(
        dif_of_projected.coeffs,
        projected_dif.coeffs,
        epsilon = 1e-12
      );
    }
  }
}
