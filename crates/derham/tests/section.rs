//! The `Section`/`Sampler` bridge: located sampling agrees with the linear
//! scan, pullback then sample is the identity on a full-dimension mesh, the
//! flat pullback is the identity special case of the curved one, the
//! composite pullback is functorial, and the field-level Hodge star is an
//! involution up to the same sign as the value-level one.

extern crate nalgebra as na;

use approx::assert_relative_eq;
use derham::Cochain;
use derham::interpolate::interpolant::WhitneyInterpolant;
use derham::project::derham_map;
use derham::section::{CoordFieldExt, Section, SectionExt, SectionOps};
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

/// The locator-accelerated sampling agrees exactly with the linear-scan
/// fallback: same cell, same reconstructed value.
#[test]
fn located_sampling_matches_scan() {
  for dim in (1..=3).into_iter().map(Dim::from) {
    let (topology, coords) = CartesianGrid::new_unit(dim, 3).triangulate();
    let locator = PointLocator::new(&topology, &coords);

    let field = DiffFormClosure::one_form(|p| p.vector().clone(), dim);
    let cochain = derham_map(&field.pullback_on(&topology, &coords), &topology, 2);
    let whitney = WhitneyInterpolant::new(cochain, &topology);

    let scan = whitney.sampled_on(&topology, &coords);
    let fast = whitney
      .sampled_on(&topology, &coords)
      .with_locator(&locator);

    for x in probe_points(dim, 4) {
      let a = scan.at_global(&x).unwrap();
      let b = fast.at_global(&x).unwrap();
      assert_relative_eq!(a.components(), b.components(), epsilon = 1e-12);
    }
  }
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

/// The flat pullback is the identity special case of the curved one: on a
/// mesh of full intrinsic dimension, `pullback_on` and
/// `pullback_through(&identity)` agree pointwise. This is the "a flat domain
/// is a continuum whose chart is the identity" claim, made a theorem.
#[test]
fn flat_pullback_is_identity_chart() {
  use glatt::parametrization::Parametrization;
  use simplicial::atlas::MeshPoint;

  for dim in (1..=3).into_iter().map(Dim::from) {
    let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();

    let field = DiffFormClosure::one_form(
      |p| Vector::from_iterator(p.dim(), p.iter().map(|x| (2.0 * x).cos())),
      dim,
    );
    let identity = Parametrization::identity(dim);

    let flat = field.pullback_on(&topology, &coords);
    let through = field.pullback_through(&topology, &coords, &identity);

    for cell in topology.cells().handle_iter() {
      let point = MeshPoint::barycenter(cell.idx());
      assert_relative_eq!(
        flat.at(&point).components(),
        through.at(&point).components(),
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

/// $star star = (-1)^(k(n-k))$ holds pointwise on the field level, with the
/// cell metric supplied by the edge lengths.
#[test]
fn hodge_star_field_involution() {
  use multiindex::Sign;
  use simplicial::atlas::MeshPoint;

  for dim in (1..=3).into_iter().map(Dim::from) {
    let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
    let lengths = coords.to_edge_lengths_sq(&topology);

    for grade in dim.range_inclusive() {
      let ndofs = topology.nsimplices(grade);
      let cochain = Cochain::new(
        grade,
        Vector::from_iterator(ndofs, (0..ndofs).map(|i| (i % 5) as f64 - 2.0)),
      );
      let orientation = topology.orientation().unwrap();
      let whitney = WhitneyInterpolant::new(cochain, &topology);
      let star_star = WhitneyInterpolant::new(whitney.cochain().clone(), &topology)
        .hodge_star(&topology, &lengths, orientation)
        .hodge_star(&topology, &lengths, orientation);

      let sign = Sign::from_parity(grade.index() * (dim - grade).index());
      for cell in topology.cells().handle_iter() {
        let point = MeshPoint::barycenter(cell.idx());
        assert_relative_eq!(
          star_star.at(&point).components(),
          &(sign.as_f64() * whitney.at(&point)).components(),
          epsilon = 1e-12
        );
      }
    }
  }
}
