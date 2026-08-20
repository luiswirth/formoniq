//! The de Rham map: it is a cochain map ($R compose dif = dif compose R$,
//! Stokes' theorem), and it does not depend on which cell supports a face.

extern crate nalgebra as na;

use approx::assert_relative_eq;
use coorder::Coord;
use derham::project::{derham_map, integrate_face};
use derham::section::CoordFieldExt;
use glatt::field::DiffFormClosure;
use multialgebra::Tensor;
use multiindex::Dim;
use regge::mesher::cartesian::CartesianGrid;
use simplicial::atlas::SimplexQuadRule;
use simplicial::linalg::Vector;

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
      DiffFormClosure::scalar(|p| p[0] * p[0], Dim::new(1)),
      DiffFormClosure::one_form(|p| Vector::from_element(1, 2.0 * p[0]), Dim::new(1)),
    ),
    // 2d: omega = x y, dif omega = y dx + x dy
    (
      DiffFormClosure::scalar(|p| p[0] * p[1], Dim::new(2)),
      DiffFormClosure::one_form(|p| na::dvector![p[1], p[0]], Dim::new(2)),
    ),
    // 2d: omega = y dx, dif omega = -dx wedge dy
    (
      DiffFormClosure::one_form(|p| na::dvector![p[1], 0.0], Dim::new(2)),
      DiffFormClosure::new(
        |_| Tensor::multiform(na::dvector![-1.0], Dim::new(2), Dim::new(2)),
        Dim::new(2),
        Dim::new(2),
      ),
    ),
    // 3d: omega = z dy, dif omega = -dy wedge dz
    (
      DiffFormClosure::one_form(|p| na::dvector![0.0, p[2], 0.0], Dim::new(3)),
      DiffFormClosure::new(
        |_| Tensor::multiform(na::dvector![0.0, 0.0, -1.0], Dim::new(3), Dim::new(2)),
        Dim::new(3),
        Dim::new(2),
      ),
    ),
    // 3d: omega = x dy wedge dz, dif omega = dx wedge dy wedge dz
    (
      DiffFormClosure::new(
        |p| Tensor::multiform(na::dvector![0.0, 0.0, p[0]], Dim::new(3), Dim::new(2)),
        Dim::new(3),
        Dim::new(2),
      ),
      DiffFormClosure::new(
        |_| Tensor::multiform(na::dvector![1.0], Dim::new(3), Dim::new(3)),
        Dim::new(3),
        Dim::new(3),
      ),
    ),
  ];

  for (form, dif_form) in cases {
    let dim = glatt::field::CoordField::dim(&form);
    let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();

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
  for dim in (2..=3).into_iter().map(Dim::from) {
    let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
    let field = DiffFormClosure::one_form(
      |p: &Coord| Vector::from_iterator(p.dim(), p.iter().map(|x| x.sin())),
      dim,
    );
    let pulled = field.pullback_on(&topology, &coords);
    let qr = SimplexQuadRule::degree(Dim::ONE, 3);

    for edge in topology.skeleton(Dim::ONE).handle_iter() {
      let integrals: Vec<f64> = edge
        .cells()
        .map(|cell| {
          let positions = edge.simplex().relative_to(cell.simplex());
          integrate_face(&pulled, cell, &positions, &qr)
        })
        .collect();

      for value in &integrals[1..] {
        assert_relative_eq!(*value, integrals[0], epsilon = 1e-12);
      }
    }
  }
}
