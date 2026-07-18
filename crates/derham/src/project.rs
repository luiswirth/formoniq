//! The de Rham map $R: L^2 Lambda^k -> C^k$.
//!
//! Discretization of differential forms into cochains by integration over
//! the simplices of the mesh. Together with the Whitney interpolation
//! $W: C^k -> L^2 Lambda^k$ (see [`crate::interpolate`]) it forms the pair of
//! cochain maps at the heart of FEEC. The governing laws are executable:
//!
//! - $R compose W = id$: Whitney's theorem
//!   (test `whitney_basis_property` in [`crate`]).
//! - $R compose dif = dif compose R$: Stokes' theorem
//!   (test `derham_map_is_cochain_map`, below).
//! - $dif compose W = W compose dif$: Whitney forms are a subcomplex
//!   (test `whitney_interpolation_is_cochain_map` in
//!   [`crate::interpolate::interpolant`]).
//!
//! The integral $integral_sigma omega$ of a $k$-form over a $k$-simplex is
//! **metric-free** -- it pairs the form with the tangent blade of the simplex,
//! and no length, angle or volume is ever needed. The implementation is
//! correspondingly intrinsic, and needs no embedding either: it works entirely
//! in the chart of a cell supporting $sigma$, where the face is an affine
//! subsimplex of the reference cell and its tangent blade is pure
//! combinatorics.
//!
//! Which supporting cell is chosen does not matter. Two charts containing
//! $sigma$ differ by a [`Transition`](simplicial::atlas::Transition), whose
//! differential carries the tangent blade of $sigma$ in one chart to the tangent
//! blade in the other; the pairing sees only the part of $omega$ tangential to
//! $sigma$, on which the two charts agree by exactly that map. The law is
//! `derham_map_is_independent_of_supporting_cell` below.

use crate::{cochain::Cochain, section::Section};

use {
  exterior::{exterior_power, Covariant, MultiVector},
  multiindex::Combination,
  simplicial::{
    atlas::{ref_face_spanning_vectors, refsimp_vol, MeshPoint, SimplexQuadRule},
    topology::{complex::Complex, handle::SimplexIdx},
    Dim,
  },
};

/// The de Rham map: discretize a differential $k$-form on the simplicial
/// manifold into a $k$-cochain by integrating it over each $k$-simplex, with
/// quadrature exact for polynomial integrands of the given degree.
///
/// Metric-free, and defined on any geometry -- including none at all. An
/// analytic form given in coordinates reaches this through the pullback, so
/// that $R (phi^* omega)$ reads as the composition it is:
///
/// ```ignore
/// derham_map(&omega.pullback_on(&topology, &coords), &topology, 1)
/// ```
pub fn derham_map(
  field: &impl Section<Covariant>,
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
      integrate_over_face(field, cell.idx(), &positions, &qr)
    })
    .collect::<Vec<_>>()
    .into();

  Cochain::new(grade, coeffs)
}

/// The tangent blade $v_1 wedge dots.c wedge v_k$ of a face of the reference
/// cell, in the cell's reference frame: the single column
/// $Lambda^k V in RR^(binom(n,k) times 1)$ of the $k$-minors of its spanning
/// vectors.
///
/// An `exterior` construction on `simplicial` combinatorics, and so it lives here
/// in the crate that joins them. Metric-free and coordinate-free: a face of a
/// cell has spanning vectors in the cell's chart whatever geometry the mesh
/// carries, and none at all if it carries none.
pub fn face_tangent_blade(cell_dim: Dim, positions: &Combination) -> MultiVector {
  let grade = positions.card() - 1;
  let spanning = ref_face_spanning_vectors(cell_dim, positions);
  let coeffs = exterior_power(&spanning, grade).column(0).into_owned();
  MultiVector::new(coeffs, cell_dim, grade)
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
  field: &impl Section<Covariant>,
  cell: SimplexIdx,
  positions: &Combination,
  qr: &SimplexQuadRule,
) -> f64 {
  let grade = positions.card() - 1;
  assert_eq!(qr.dim(), grade);
  assert_eq!(field.grade(), grade);

  let tangent_blade = face_tangent_blade(cell.dim(), positions);
  let integrand = |point: &MeshPoint| field.at(point).pairing(&tangent_blade);
  qr.integrate_face(cell, positions, &integrand, refsimp_vol(grade))
}

#[cfg(test)]
mod test {
  use super::*;

  use crate::section::CoordFieldExt;

  use {
    chartan::field::DiffFormClosure, coorder::Coord, exterior::ExteriorElement,
    simplicial::gen::cartesian::CartesianMeshInfo, simplicial::linalg::Vector,
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
      let dim = chartan::field::CoordField::dim(&form);
      let (topology, coords) = CartesianMeshInfo::new_unit(dim, 2).compute_coord_complex();

      let dif_of_projected =
        derham_map(&form.pullback_on(&topology, &coords), &topology, 1).dif(&topology);
      let projected_dif = derham_map(&dif_form.pullback_on(&topology, &coords), &topology, 1);

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
    for dim in 2..=3 {
      let (topology, coords) = CartesianMeshInfo::new_unit(dim, 2).compute_coord_complex();
      let field = DiffFormClosure::one_form(
        |p: &Coord| Vector::from_iterator(p.dim(), p.iter().map(|x| x.sin())),
        dim,
      );
      let pulled = field.pullback_on(&topology, &coords);
      let qr = SimplexQuadRule::degree(1, 3);

      for edge in topology.skeleton(1).handle_iter() {
        let integrals: Vec<f64> = edge
          .cells()
          .map(|cell| {
            let positions = edge.simplex().relative_to(cell.simplex());
            integrate_over_face(&pulled, cell.idx(), &positions, &qr)
          })
          .collect();

        for value in &integrals[1..] {
          assert_relative_eq!(*value, integrals[0], epsilon = 1e-12);
        }
      }
    }
  }

  /// The tangent blade of a shared face transforms by $Lambda^k (dif psi)$
  /// under the transition between the two charts that see it.
  ///
  /// This is *why* the de Rham map is well defined: the duality pairing
  /// $angle.l omega, tau angle.r$ is invariant because the form pulls back along
  /// $dif psi$ exactly as the blade pushes forward along it, and the two cancel.
  /// The well-definedness above is the consequence; this is the cause.
  #[test]
  fn tangent_blade_transforms_by_the_transition_differential() {
    use simplicial::atlas::ChartExt;

    for dim in 2..=3 {
      let (topology, _) = CartesianMeshInfo::new_unit(dim, 2).compute_coord_complex();

      for face_dim in 1..dim {
        for face in topology.skeleton(face_dim).handle_iter() {
          let cells: Vec<_> = face.cells().collect();
          for (i, &source) in cells.iter().enumerate() {
            for &target in &cells[i + 1..] {
              let differential = source.chart().transition_to(target.chart()).differential();

              let here = face_tangent_blade(dim, &face.simplex().relative_to(source.simplex()));
              let there = face_tangent_blade(dim, &face.simplex().relative_to(target.simplex()));

              let pushed = here.pushforward(&differential);
              assert_relative_eq!(pushed.coeffs(), there.coeffs(), epsilon = 1e-12);
            }
          }
        }
      }
    }
  }
}
