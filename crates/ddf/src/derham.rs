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
//!   (test `whitney_interpolation_is_cochain_map` in [`crate::whitney::form`]).
//!
//! The integral $integral_sigma omega$ of a $k$-form over a $k$-simplex is
//! **metric-free** -- it pairs the form with the tangent blade of the simplex,
//! and no length, angle or volume is ever needed. The implementation is
//! correspondingly intrinsic: it works entirely in the reference frame of a
//! cell supporting $sigma$, where the face is an affine subsimplex of the
//! standard cell and its tangent blade is pure combinatorics. Which supporting
//! cell is chosen does not matter: neighbouring charts differ by an affine
//! gluing that fixes $sigma$, and the pairing sees only the part of $omega$
//! tangential to $sigma$, on which the charts agree.

use crate::{field::ExteriorField, CoordSimplexExt};

use {
  common::combo::Combination,
  exterior::Covariant,
  manifold::{
    geometry::{
      coord::{mesh::MeshCoords, quadrature::SimplexQuadRule, simplex::SimplexCoords, CoordRef},
      refsimp_vol,
    },
    point::MeshPoint,
    topology::{complex::Complex, handle::SimplexIdx, simplex::Simplex},
    Dim,
  },
};

use crate::cochain::Cochain;

/// The de Rham map: discretize a differential $k$-form on the manifold into a
/// $k$-cochain by integrating it over each $k$-simplex of the mesh, with
/// quadrature exact for polynomial integrands of the given degree.
///
/// Metric-free, and defined on any geometry -- including none at all.
pub fn derham_map(
  field: &impl ExteriorField<Covariant>,
  topology: &Complex,
  quad_degree: usize,
) -> Cochain {
  let grade = field.grade();
  let qr = SimplexQuadRule::degree(grade, quad_degree);

  let coeffs = topology
    .skeleton(grade)
    .handle_iter()
    .map(|simp| {
      let cell = simp.cells().next().expect("Every simplex has a cell.");
      let positions = simp.simplex().relative_to(cell.simplex());
      let face = reference_face(topology.dim(), &positions);
      integrate_over_face(field, cell.idx(), &face, &qr)
    })
    .collect::<Vec<_>>()
    .into();

  Cochain::new(grade, coeffs)
}

/// The de Rham map of a coordinate form, pulled back onto the mesh.
pub fn derham_map_coord(
  field: &impl exterior::field::CoordField<Covariant>,
  topology: &Complex,
  coords: &MeshCoords,
  quad_degree: usize,
) -> Cochain {
  use crate::field::CoordFieldExt;
  derham_map(&field.pullback_on(topology, coords), topology, quad_degree)
}

/// The face of the reference $n$-simplex spanned by the given local vertex
/// positions: the image of a face of a cell under that cell's chart.
///
/// A coordinate simplex, but not an embedding of the mesh -- these are the
/// coordinates *of the chart*, which every cell has by definition.
pub fn reference_face(cell_dim: Dim, positions: &Combination) -> SimplexCoords {
  let simp = Simplex::new(positions.iter().collect());
  SimplexCoords::from_simplex_and_coords(&simp, &MeshCoords::standard(cell_dim))
}

/// $integral_sigma omega$ over a face of a cell, expressed in that cell's
/// reference frame.
///
/// The pullback of $omega$ to the reference $k$-simplex is
/// $angle.l omega, v_1 wedge dots.c wedge v_k angle.r dif x^1 wedge dots.c wedge dif x^k$
/// for the spanning vectors $v_i$ of the face, so the integral is the
/// quadrature of the duality pairing against the face's tangent blade --
/// no metric anywhere.
pub fn integrate_over_face(
  field: &impl ExteriorField<Covariant>,
  cell: SimplexIdx,
  face: &SimplexCoords,
  qr: &SimplexQuadRule,
) -> f64 {
  let grade = face.dim_intrinsic();
  assert_eq!(qr.dim(), grade);
  assert_eq!(field.grade(), grade);

  let tangent_blade = face.spanning_multivector();
  let integrand = |local: CoordRef| {
    let point = MeshPoint::from_local(cell, face.local2global(local).as_view());
    field.at(&point).pairing(&tangent_blade)
  };
  qr.integrate_local(&integrand, refsimp_vol(grade))
}

#[cfg(test)]
mod test {
  use super::*;

  use {
    common::linalg::nalgebra::Vector,
    exterior::{field::DiffFormClosure, ExteriorElement},
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
        DiffFormClosure::new(|_| ExteriorElement::new(na::dvector![-1.0], 2, 2), 2, 2),
      ),
      // 3d: omega = z dy, dif omega = -dy wedge dz
      (
        DiffFormClosure::one_form(|p| na::dvector![0.0, p[2], 0.0], 3),
        DiffFormClosure::new(
          |_| ExteriorElement::new(na::dvector![0.0, 0.0, -1.0], 3, 2),
          3,
          2,
        ),
      ),
      // 3d: omega = x dy wedge dz, dif omega = dx wedge dy wedge dz
      (
        DiffFormClosure::new(
          |p| ExteriorElement::new(na::dvector![0.0, 0.0, p[0]], 3, 2),
          3,
          2,
        ),
        DiffFormClosure::new(|_| ExteriorElement::new(na::dvector![1.0], 3, 3), 3, 3),
      ),
    ];

    for (form, dif_form) in cases {
      let dim = exterior::field::CoordField::dim(&form);
      let (topology, coords) = CartesianMeshInfo::new_unit(dim, 2).compute_coord_complex();

      let dif_of_projected = derham_map_coord(&form, &topology, &coords, 1).dif(&topology);
      let projected_dif = derham_map_coord(&dif_form, &topology, &coords, 1);

      assert_eq!(dif_of_projected.grade(), projected_dif.grade());
      assert_relative_eq!(
        dif_of_projected.coeffs(),
        projected_dif.coeffs(),
        epsilon = 1e-12
      );
    }
  }

  /// The de Rham map does not depend on which cell supports a face: a form
  /// integrated over an interior simplex gives the same number from either
  /// side of it.
  ///
  /// This is the well-definedness of $R$ on the manifold, and it is what makes
  /// the reference-frame implementation legitimate.
  #[test]
  fn derham_map_is_independent_of_supporting_cell() {
    use crate::field::CoordFieldExt;

    for dim in 2..=3 {
      let (topology, coords) = CartesianMeshInfo::new_unit(dim, 2).compute_coord_complex();
      let field = DiffFormClosure::one_form(
        |p| Vector::from_iterator(p.len(), p.iter().map(|x| x.sin())),
        dim,
      );
      let pulled = field.pullback_on(&topology, &coords);
      let qr = SimplexQuadRule::degree(1, 3);

      for edge in topology.skeleton(1).handle_iter() {
        let integrals: Vec<f64> = edge
          .cells()
          .map(|cell| {
            let positions = edge.simplex().relative_to(cell.simplex());
            let face = reference_face(dim, &positions);
            integrate_over_face(&pulled, cell.idx(), &face, &qr)
          })
          .collect();

        for value in &integrals[1..] {
          assert_relative_eq!(*value, integrals[0], epsilon = 1e-12);
        }
      }
    }
  }
}
