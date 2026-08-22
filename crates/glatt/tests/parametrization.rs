//! [`Parametrization`]: $chi compose phi = id$, the chart being the inverse of
//! the forward map, and $phi$ landing on the shape its implicit equation cuts
//! out. Swept over every shape the library states it over (sphere, ball,
//! torus, graph), and the round metric of the unit sphere as an anchor.
//!
//! The graph is the only shape with no closed-form chart, so it is the one
//! carrying the fully derived path, the finite-difference Jacobian and the
//! Gauss-Newton nearest-point solve.

extern crate nalgebra as na;

use approx::assert_relative_eq;
use coorder::{Coord, Matrix};
use glatt::parametrization::Parametrization;
use multialgebra::Dim;

/// A shape together with the data the law is stated against: the parameter
/// points where the chart is a genuine inverse, and the residual of the
/// implicit equation cutting the image out of the ambient space.
struct Shape {
  name: String,
  map: Parametrization,
  probes: Vec<Coord>,
  residual: Box<dyn Fn(&Coord) -> f64>,
}

impl Shape {
  fn new(
    name: impl Into<String>,
    map: Parametrization,
    probes: impl IntoIterator<Item = Vec<f64>>,
    residual: impl Fn(&Coord) -> f64 + 'static,
  ) -> Self {
    let probes = probes
      .into_iter()
      .map(|u| Coord::from_iterator(u.len(), u))
      .collect();
    Self {
      name: name.into(),
      map,
      probes,
      residual: Box::new(residual),
    }
  }
}

/// The shapes, over the dimensions each is stated over.
///
/// The parameter points stay strictly inside the coordinate box, where the
/// chart is a genuine inverse: the azimuth returns in $(-pi, pi]$ from
/// `atan2` and the polar angles in $(0, pi)$, so the round trip is exact only
/// there. The graph is the exception and is probed on an extended domain,
/// having no periodic coordinate.
fn shapes() -> Vec<Shape> {
  let mut shapes = Vec::new();

  for dim in 1..=4 {
    let radius = 0.5 + 0.5 * f64::from(dim);
    let angles = (0..dim)
      .map(|k| {
        if k + 1 == dim {
          2.0
        } else {
          0.3 + 0.7 * f64::from(k)
        }
      })
      .collect::<Vec<_>>();
    shapes.push(Shape::new(
      format!("sphere({dim}, {radius})"),
      Parametrization::sphere(Dim::from(dim), radius),
      [angles],
      move |p| p.norm() - radius,
    ));
  }

  for dim in 2..=4 {
    let mut coords = vec![1.3]; // radius, inside (0, 2)
    coords.extend((0..dim - 1).map(|k| {
      if k + 2 == dim {
        1.5
      } else {
        0.4 + 0.6 * f64::from(k)
      }
    }));
    let radius = coords[0];
    shapes.push(Shape::new(
      format!("ball({dim})"),
      Parametrization::ball(Dim::from(dim)),
      [coords],
      // The ball is solid, so the image fills an open set: the residual is the
      // radial coordinate the chart has to return, not a constraint.
      move |p| p.norm() - radius,
    ));
  }

  let (major, minor) = (3.0, 1.0);
  shapes.push(Shape::new(
    "torus",
    Parametrization::torus(major, minor),
    [vec![0.3, 0.7], vec![2.0, -1.1], vec![-2.5, 3.0]],
    move |p| {
      let rho = p[0].hypot(p[1]);
      (rho - major).powi(2) + p[2] * p[2] - minor * minor
    },
  ));

  // A genuinely curved height, so the vertical drop is only an approximate
  // seed and Gauss-Newton has to do real work.
  let height = |u: &Coord| 0.4 * u.iter().map(|x| x.sin()).sum::<f64>();
  for dim in 1..=3 {
    let probes = (0..5)
      .map(|step| (0..dim).map(|k| -0.8 + 0.5 * (k + step) as f64).collect())
      .collect::<Vec<Vec<f64>>>();
    shapes.push(Shape::new(
      format!("graph({dim})"),
      Parametrization::graph(height, Dim::from(dim)),
      probes,
      move |p| {
        let base = Coord::from_iterator(dim, p.vector().rows(0, dim).iter().copied());
        p[dim] - height(&base)
      },
    ));
  }

  shapes
}

/// $chi compose phi = id$ and $phi(u)$ lies on the shape: the chart is the
/// inverse of the parametrization, and the forward map lands in the zero set
/// of the implicit equation.
///
/// One law over every shape, so the closed-form charts and the derived
/// Gauss-Newton one are held to the same statement.
#[test]
fn the_chart_inverts_the_parametrization_onto_the_shape() {
  for shape in shapes() {
    let Shape {
      name,
      map,
      probes,
      residual,
    } = shape;
    for u in probes {
      let p = map.forward(&u);
      assert!(
        residual(&p).abs() < 1e-12,
        "{name}: the image is off the shape"
      );

      let recovered = map.chart(&p, &map.seed_at(&p));
      assert!(
        (recovered.vector() - u.vector()).norm() < 1e-9,
        "{name}: the chart did not invert the parametrization"
      );
    }
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
