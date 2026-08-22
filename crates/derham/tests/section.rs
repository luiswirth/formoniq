//! The pullback bridge between the continuum and the simplicial manifold:
//! it is functorial, and on a mesh of full intrinsic dimension it is
//! invertible by the sampler.

extern crate nalgebra as na;

use approx::assert_relative_eq;
use derham::section::{CoordFieldExt, Section, SectionExt};
use glatt::field::{CoordField, DiffFormClosure};
use multiindex::Dim;
use regge::coord::Coord;
use regge::coord::locate::PointLocator;
use regge::mesher::cartesian::CartesianGrid;
use simplicial::linalg::Vector;

/// A grid of sample points, kept off cell faces and triangulation diagonals.
fn probe_points(dim: Dim, samples: usize) -> impl Iterator<Item = Coord> {
  let phase = [0.017, 0.113, 0.237];
  (0..samples.pow(dim.index() as u32)).map(move |flat| {
    Coord::from_iterator(
      dim.index(),
      (0..dim.index()).map(|d| {
        let i = flat / samples.pow(d as u32) % samples;
        0.05 + 0.9 * (i as f64 + phase[d]) / samples as f64
      }),
    )
  })
}

/// Pulling a coordinate form onto the mesh and sampling it back in ambient
/// coordinates is the identity, on a mesh of full intrinsic dimension.
///
/// There the chart is invertible, so the pseudo-inverse is the inverse and
/// $((A^+)^* compose A^*) omega = omega$, the round trip through the
/// manifold loses nothing.
#[test]
fn pullback_then_sample_is_identity() {
  for dim in (1..=3).into_iter().map(Dim::from) {
    let (topology, coords) = CartesianGrid::new_unit(dim, 3).triangulate();
    let locator = PointLocator::new(&topology, &coords);

    let field = DiffFormClosure::one_form(
      |p| Vector::from_iterator(p.dim(), p.iter().map(|x| (2.0 * x).sin())),
      dim,
    );
    let pulled = field.pullback_on(&topology, &coords);
    let sampled = pulled.sampled_on(&topology, &coords).with_locator(&locator);

    for x in probe_points(dim, 4) {
      assert_relative_eq!(
        sampled.at_global(&x).unwrap().components(),
        field.at(&x).components(),
        epsilon = 1e-12
      );
    }
  }
}

/// Functoriality of the composite differential: pulling a continuum form onto
/// a curved mesh through $chi compose psi_K$ equals pulling it along
/// $dif chi$ and then along $dif psi_K$ separately, $Lambda^k (A B) =
/// (Lambda^k A)(Lambda^k B)$, exercised through the real bridge on $S^2$.
///
/// The bridge assembles the single composite $dif chi dot dif psi_K$; the
/// two-stage pullback here brackets it the other way. Both sides evaluate the
/// same finite-difference Jacobian, so the check is deterministic.
#[test]
fn composite_pullback_is_functorial() {
  use glatt::parametrization::Parametrization;
  use regge::{coord::simplex::SimplexRefExt, mesher::sphere::mesh_sphere_surface};
  use simplicial::atlas::MeshPoint;

  let (topology, coords) = mesh_sphere_surface(2);
  let sphere = Parametrization::sphere(Dim::new(2), 1.0);

  // An arbitrary 1-form on the (theta, phi) chart domain.
  let form =
    DiffFormClosure::one_form(|u| na::dvector![u[0].sin(), u[0] * u[1].cos()], Dim::new(2));
  let section = form.pullback_through(&topology, &coords, &sphere);

  for cell in topology.cells().handle_iter() {
    let point = MeshPoint::barycenter(cell.idx());
    let parametrization = cell.coord_simplex(&coords);
    let global = parametrization.bary2global(point.bary());
    let u = sphere.chart(&global, sphere.seed());
    let dchi = sphere.chart_differential(&u);
    let dpsi = parametrization.linear_transform();

    let staged = form.at(&u).pullback(&dchi).pullback(&dpsi);
    assert_relative_eq!(
      section.at(&point).components(),
      staged.components(),
      epsilon = 1e-12
    );
  }
}
