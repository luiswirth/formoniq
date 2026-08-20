//! [`Parametrization`]: the finite-difference Jacobian matches the
//! analytic one, the Gauss-Newton chart inverts the forward map (and
//! terminates early off the manifold), and the built-in shapes (sphere,
//! ball, torus, graph) each satisfy their own closed-form check.

extern crate nalgebra as na;

use approx::assert_relative_eq;
use coorder::{Coord, Matrix};
use glatt::parametrization::{GN_MAX_ITER, Parametrization};
use multialgebra::Dim;

/// $(theta, phi) |-> RR^3$: the unit sphere in spherical coordinates, with a
/// closed-form inverse to test the derived machinery against.
fn sphere() -> Parametrization {
  Parametrization::new(
    |u: &Coord| {
      let (theta, phi) = (u[0], u[1]);
      Coord::from_iterator(
        3,
        [
          theta.sin() * phi.cos(),
          theta.sin() * phi.sin(),
          theta.cos(),
        ],
      )
    },
    Dim::new(2),
  )
}

/// The finite-difference Jacobian matches the analytic one, column by column.
#[test]
fn fd_jacobian_matches_analytic() {
  let sphere = sphere();
  for &(theta, phi) in &[(0.7, 0.3), (1.2, 2.1), (2.4, 5.0)] {
    let u = Coord::from_iterator(2, [theta, phi]);
    let analytic = Matrix::from_columns(&[
      na::dvector![
        theta.cos() * phi.cos(),
        theta.cos() * phi.sin(),
        -theta.sin()
      ],
      na::dvector![-theta.sin() * phi.sin(), theta.sin() * phi.cos(), 0.0],
    ]);
    assert_relative_eq!(sphere.jacobian(&u), analytic, epsilon = 1e-6);
  }
}

/// $chi compose phi = id$ on $Omega$: the Gauss-Newton chart inverts the
/// forward map. Seeded near the point, since the sphere's $phi$ is not
/// injective globally.
#[test]
fn chart_inverts_forward() {
  let sphere = sphere();
  for &(theta, phi) in &[(0.7, 0.3), (1.2, 2.1), (2.4, 5.0)] {
    let u = Coord::from_iterator(2, [theta, phi]);
    let p = sphere.forward(&u);
    let seed = Coord::from_iterator(2, [theta + 0.1, phi - 0.1]);
    let recovered = sphere.chart(&p, &seed);
    assert_relative_eq!(recovered.vector(), u.vector(), epsilon = 1e-9);
  }
}

/// The Gauss-Newton chart lands on the manifold when the target is off it:
/// the nearest-point projection of an inflated point returns the radial
/// footpoint.
#[test]
fn chart_projects_off_manifold() {
  let sphere = sphere();
  let u = Coord::from_iterator(2, [1.0, 2.0]);
  let footpoint = sphere.forward(&u);
  let inflated = Coord::new(footpoint.vector() * 1.3);
  let recovered = sphere.chart(&inflated, &u);
  assert_relative_eq!(recovered.vector(), u.vector(), epsilon = 1e-9);
}

/// Gauss-Newton stops once it has converged, on a target off the manifold as
/// much as on one on it. The residual there is the distance from the point to
/// the image and never vanishes, so what converges is its tangential part,
/// the step, and a solver watching the residual would run to the iteration
/// cap on every such query while returning the same answer.
#[test]
fn the_projection_of_an_off_manifold_point_terminates_early() {
  use std::sync::Arc;
  use std::sync::atomic::{AtomicUsize, Ordering};

  let evaluations = Arc::new(AtomicUsize::new(0));
  let counter = evaluations.clone();
  let sphere = Parametrization::new(
    move |u: &Coord| {
      counter.fetch_add(1, Ordering::Relaxed);
      let (theta, phi) = (u[0], u[1]);
      Coord::from_iterator(
        3,
        [
          theta.sin() * phi.cos(),
          theta.sin() * phi.sin(),
          theta.cos(),
        ],
      )
    },
    Dim::new(2),
  );

  let u = Coord::from_iterator(2, [1.0, 2.0]);
  let inflated = Coord::new(sphere.forward(&u).vector() * 1.3);
  evaluations.store(0, Ordering::Relaxed);
  let recovered = sphere.chart(&inflated, &u);

  assert_relative_eq!(recovered.vector(), u.vector(), epsilon = 1e-9);
  // Every iteration evaluates the forward map at least once, so staying
  // under the cap in evaluations is staying under it in iterations.
  // Every iteration evaluates the forward map at least once, so staying
  // under the cap in evaluations is staying under it in iterations.
  assert!(evaluations.load(Ordering::Relaxed) < GN_MAX_ITER);
}

/// The built-in $n$-sphere, across dimensions and radii: every image lies on
/// the sphere of the requested radius, and its closed-form chart inverts the
/// forward map on the angle box regardless of radius.
#[test]
fn hypersphere_has_the_right_radius_and_chart() {
  for dim in 1..=4 {
    let radius = 0.5 + 0.5 * dim as f64;
    let sphere = Parametrization::sphere(dim.into(), radius);
    // Angles strictly inside the box, where the chart is a genuine inverse.
    // The azimuth returns in $(-pi, pi]$ from `atan2`; keep it there so the
    // round-trip is exact, and the polar angles in $(0, pi)$.
    let angles: Vec<f64> = (0..dim)
      .map(|k| {
        if k + 1 == dim {
          2.0
        } else {
          0.3 + 0.7 * k as f64
        }
      })
      .collect();
    let u = Coord::from_iterator(dim, angles);

    let p = sphere.forward(&u);
    assert_eq!(p.dim(), dim + 1);
    assert_relative_eq!(p.norm(), radius, epsilon = 1e-12);

    let recovered = sphere.chart(&p, sphere.seed());
    assert_relative_eq!(recovered.vector(), u.vector(), epsilon = 1e-12);
  }
}

/// The graph validates the fully derived path: with no closed-form chart, the
/// finite-difference Jacobian and Gauss-Newton chart (seeded by the vertical
/// drop) invert the forward map across an extended domain, and the image sits
/// on the graph. Swept over dimensions.
#[test]
fn graph_derived_chart_inverts() {
  for dim in 1..=3 {
    // A genuinely curved height, so the vertical drop is only an approximate
    // seed and Gauss-Newton has to do real work.
    let height = |u: &Coord| 0.4 * u.iter().map(|x| x.sin()).sum::<f64>();
    let graph = Parametrization::graph(height, dim.into());

    for step in 0..5 {
      let u = Coord::from_iterator(dim, (0..dim).map(|k| -0.8 + 0.5 * (k + step) as f64));
      let p = graph.forward(&u);
      assert_eq!(p.dim(), dim + 1);
      assert_relative_eq!(p[dim], height(&u), epsilon = 1e-12);

      let recovered = graph.chart(&p, &graph.seed_at(&p));
      assert_relative_eq!(recovered.vector(), u.vector(), epsilon = 1e-9);
    }
  }
}

/// The solid ball, across dimensions: the forward image has the requested
/// radial coordinate as its norm, and the closed-form chart inverts it. The
/// metric is flat, so this exercises the curvilinear chart alone.
#[test]
fn ball_chart_inverts() {
  for dim in 2..=4 {
    let ball = Parametrization::ball(dim.into());
    let mut coords = vec![1.3]; // radius, inside (0, 2)
    coords.extend((0..dim - 1).map(|k| {
      if k + 2 == dim {
        1.5
      } else {
        0.4 + 0.6 * k as f64
      }
    }));
    let u = Coord::from_iterator(dim, coords);

    let p = ball.forward(&u);
    assert_eq!(p.dim(), dim);
    assert_relative_eq!(p.norm(), u[0], epsilon = 1e-12);

    let recovered = ball.chart(&p, ball.seed());
    assert_relative_eq!(recovered.vector(), u.vector(), epsilon = 1e-12);
  }
}

/// The 2-torus: every image satisfies the implicit torus equation
/// $(sqrt(x^2 + y^2) - R)^2 + z^2 = r^2$, and the toroidal-angle chart inverts
/// the forward map.
#[test]
fn torus_lies_on_surface_and_chart_inverts() {
  let (major, minor) = (3.0, 1.0);
  let torus = Parametrization::torus(major, minor);
  for &(theta, phi) in &[(0.3, 0.7), (2.0, -1.1), (-2.5, 3.0)] {
    let u = Coord::from_iterator(2, [theta, phi]);
    let p = torus.forward(&u);
    let rho = (p[0] * p[0] + p[1] * p[1]).sqrt();
    assert_relative_eq!(
      (rho - major).powi(2) + p[2] * p[2],
      minor * minor,
      epsilon = 1e-12
    );

    let recovered = torus.chart(&p, torus.seed());
    assert_relative_eq!(recovered.vector(), u.vector(), epsilon = 1e-12);
  }
}

/// The induced metric of the round unit sphere is the textbook
/// $g = "diag"(1, sin^2 theta)$: the Gramian of the tangent frame, recovered
/// from the finite-difference Jacobian alone.
#[test]
fn sphere_induced_metric_is_round() {
  let sphere = Parametrization::sphere(Dim::new(2), 1.0);
  for &(theta, phi) in &[(0.7, 0.3), (1.2, 2.1), (2.4, 5.0)] {
    let u = Coord::from_iterator(2, [theta, phi]);
    let g = sphere.induced_metric(&u);
    let expected = Matrix::from_diagonal(&na::dvector![1.0, theta.sin().powi(2)]);
    assert_relative_eq!(g.matrix(), &expected, epsilon = 1e-6);
  }
}

/// The identity parametrization is its own chart and has the identity
/// Jacobian, no solve involved.
#[test]
fn identity_is_trivial() {
  let id = Parametrization::identity(Dim::new(3));
  let p = Coord::from_iterator(3, [1.0, -2.0, 0.5]);
  assert_eq!(id.forward(&p).vector(), p.vector());
  assert_eq!(id.chart(&p, &Coord::zeros(3)).vector(), p.vector());
  assert_eq!(id.jacobian(&p), Matrix::identity(3, 3));
}
