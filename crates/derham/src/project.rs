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
//! metric-free, it pairs the form with the tangent blade of the simplex,
//! and no length, angle or volume is ever needed. The implementation is
//! correspondingly intrinsic, and needs no embedding either: it works entirely
//! in the chart of a cell supporting $sigma$, where the face is an affine
//! subsimplex of the reference cell and its tangent blade is pure
//! combinatorics.
//!
//! Which supporting cell is chosen does not matter, and the reason is a fact
//! about the atlas rather than about this map: two charts containing $sigma$
//! differ by a [`Transition`](simplicial::atlas::Transition), which carries the
//! tangential part of a fiber value faithfully and nothing else. The pairing of
//! $omega$ with the tangent blade of $sigma$ is tangential, so the two charts
//! agree on it. The law is `derham_map_is_independent_of_supporting_cell` below,
//! and its cause is stated and tested one crate down.

use crate::{Cochain, section::Section};

use {
  multiindex::Combination,
  simplicial::{
    atlas::{Chart, FaceTrace, MeshPoint, SimplexQuadRule, unit_simplex_volume},
    topology::complex::Complex,
  },
};

/// The de Rham map: discretize a differential $k$-form on the simplicial
/// manifold into a $k$-cochain by integrating it over each $k$-simplex, with
/// quadrature exact for polynomial integrands of the given degree.
///
/// Metric-free, and defined on any geometry, including none at all. An
/// analytic form given in coordinates reaches this through the pullback, so
/// that $R (phi^* omega)$ reads as the composition it is:
///
/// ```ignore
/// derham_map(&omega.pullback_on(&topology, &coords), &topology, 1)
/// ```
pub fn derham_map(field: &impl Section, topology: &Complex, quad_degree: usize) -> Cochain {
  let grade = field.grade();
  let qr = SimplexQuadRule::degree(grade, quad_degree);

  let coeffs = topology
    .skeleton(grade)
    .handle_iter()
    .map(|simp| {
      let cell = simp.cells().next().expect("Every simplex has a cell.");
      let positions = simp.simplex().relative_to(cell.simplex());
      integrate_face(field, cell, &positions, &qr)
    })
    .collect::<Vec<_>>()
    .into();

  Cochain::new(grade, coeffs)
}

/// $integral_sigma omega$ over a face of a cell, expressed in that cell's
/// reference frame.
///
/// The pullback of $omega$ to the reference $k$-simplex is
/// $chevron.l omega, v_1 wedge dots.c wedge v_k chevron.r dif x^1 wedge dots.c wedge dif x^k$
/// for the spanning vectors $v_i$ of the face, so the integral is the
/// quadrature of the duality pairing against the face's tangent blade,
/// no metric anywhere.
pub fn integrate_face(
  field: &impl Section,
  chart: Chart,
  positions: &Combination,
  qr: &SimplexQuadRule,
) -> f64 {
  let grade = positions.card() - 1;
  assert_eq!(qr.dim(), grade);
  assert_eq!(field.grade(), grade);

  // The integrand is the trace at the face's own grade: integrating a k-form
  // over a k-simplex sees only its tangential part, which is why the answer
  // does not depend on the supporting cell.
  let trace = FaceTrace::new(chart.dim(), positions, grade);
  let integrand = |point: &MeshPoint| trace.top_coefficient(&field.at(point));
  qr.integrate_face(chart, positions, &integrand, unit_simplex_volume(grade))
}
