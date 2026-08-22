//! The de Rham map $R$, which reads a differential form as a cochain by
//! integrating it over each simplex: it is a cochain map, which is Stokes'
//! theorem.

extern crate nalgebra as na;

use approx::assert_relative_eq;
use derham::project::derham_map;
use derham::section::CoordFieldExt;
use glatt::field::DiffFormClosure;
use multialgebra::Tensor;
use multiindex::Dim;
use regge::mesher::cartesian::CartesianGrid;
use simplicial::linalg::Vector;

/// $R compose dif = dif compose R$: the de Rham map is a cochain map.
///
/// This is Stokes' theorem: integrating $dif omega$ over a simplex equals
/// summing $omega$ over its boundary. Forms with affine coefficients make
/// the barycentric quadrature exact, so both sides agree up to roundoff.
#[test]
fn derham_map_is_a_cochain_map() {
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
