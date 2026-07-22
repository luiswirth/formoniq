//! The smooth parametrization of the continuum, and the chart it induces.
//!
//! The one required datum of a continuum $M$ is its **forward** map
//! $phi: Omega -> RR^N$, from a coordinate domain out into the ambient space --
//! a *parametrization*, pointing into the manifold, exactly opposite to a chart.
//! Everything else is derived, because requiring the inverse would ask for data
//! the mathematics already fixes:
//!
//! - the differential $dif phi$ (an $N times m$ matrix) by central finite
//!   difference of $phi$, unless an exact Jacobian is supplied;
//! - the chart $chi = phi^(-1) compose r$ by Gauss-Newton on
//!   $min_u norm(phi(u) - p)_2^2$, unless a closed form is supplied.
//!
//! Least-squares in the ambient Euclidean norm *is* the orthogonal (nearest
//! point) projection onto $op("im") phi$, so the derived retraction $r$ is not
//! *a* retraction but *the* geometrically optimal one -- which is what makes the
//! domain gap $O(h^2)$ rather than $O(h)$ (Dziuk, Demlow). The chart
//! differential $dif chi = (dif phi)^+$ is the pseudo-inverse of the forward
//! Jacobian: a genuine left inverse because $phi$ is an immersion.
//!
//! Metric-independent by construction: the pseudo-inverse here is the
//! Moore-Penrose one for the ambient *Euclidean* metric, which is what "nearest
//! point" means, and no `Metric` of the continuum enters.

use coorder::{Ambient, Coord, CoordSpace, Coords, Matrix, Vector};
use exterior::Dim;
use gramian::{Gramian, Metric};

/// A smooth parametrization $phi: Omega -> RR^N$ of the continuum, with its
/// derived Jacobian and chart.
///
/// Mesh-independent: it knows the continuum, not the simplicial manifold that
/// approximates it. Pulling continuum data onto a mesh through this
/// parametrization is a separate step, belonging to whatever joins the two.
pub struct Parametrization<S: CoordSpace = Ambient> {
  forward: Box<ForwardFn<S>>,
  jacobian: Option<Box<JacobianFn<S>>>,
  chart: Option<Box<ChartFn<S>>>,
  dim: Dim,
  seed: Coords<S>,
  seed_fn: Option<Box<SeedFn<S>>>,
}

/// The forward map $phi: Omega -> RR^N$.
type ForwardFn<S> = dyn Fn(&Coords<S>) -> Coord + Sync;
/// The forward Jacobian $dif phi$, an $N times m$ matrix.
type JacobianFn<S> = dyn Fn(&Coords<S>) -> Matrix + Sync;
/// The chart $chi(p, "seed") = phi^(-1)(r(p))$.
type ChartFn<S> = dyn Fn(&Coord, &Coords<S>) -> Coords<S> + Sync;
/// A Gauss-Newton seed heuristic $p |-> u_0$, from an ambient point.
type SeedFn<S> = dyn Fn(&Coord) -> Coords<S> + Sync;

/// The finite-difference step for the derived Jacobian.
const FD_STEP: f64 = 1e-7;
/// The convergence tolerance and iteration cap for the Gauss-Newton chart.
const GN_TOL: f64 = 1e-12;
const GN_MAX_ITER: usize = 100;
/// The rank tolerance of the pseudo-inverse.
const PINV_TOL: f64 = 1e-12;

impl<S: CoordSpace> Parametrization<S> {
  /// A parametrization from its forward map alone; `dim` is the dimension $m$
  /// of the domain $Omega$. The Jacobian is finite-differenced and the chart is
  /// Gauss-Newton, both seeded from the origin of $Omega$ until told otherwise.
  pub fn new(forward: impl Fn(&Coords<S>) -> Coord + Sync + 'static, dim: Dim) -> Self {
    Self {
      forward: Box::new(forward),
      jacobian: None,
      chart: None,
      dim,
      seed: Coords::zeros(dim.index()),
      seed_fn: None,
    }
  }

  /// Supply the exact Jacobian $dif phi$ ($N times m$), replacing the finite
  /// difference.
  pub fn with_jacobian(mut self, jacobian: impl Fn(&Coords<S>) -> Matrix + Sync + 'static) -> Self {
    self.jacobian = Some(Box::new(jacobian));
    self
  }

  /// Supply a closed-form chart $chi(p, "seed") = phi^(-1)(r(p))$, replacing the
  /// Gauss-Newton solve. The seed argument is ignored by an exact inverse; it is
  /// there so the two paths share a signature.
  pub fn with_chart(
    mut self,
    chart: impl Fn(&Coord, &Coords<S>) -> Coords<S> + Sync + 'static,
  ) -> Self {
    self.chart = Some(Box::new(chart));
    self
  }

  /// Set the fixed Gauss-Newton seed used where the caller has no better one.
  pub fn with_seed(mut self, seed: Coords<S>) -> Self {
    assert_eq!(seed.dim(), self.dim);
    self.seed = seed;
    self
  }

  /// Supply a seed *heuristic* $p |-> u_0$ that guesses a domain point from an
  /// ambient one, for the chart's Gauss-Newton solve. A good heuristic (e.g. the
  /// vertical drop of a graph) keeps the solve in-basin from any point, which a
  /// single fixed seed cannot across an extended domain.
  pub fn with_seed_fn(mut self, seed_fn: impl Fn(&Coord) -> Coords<S> + Sync + 'static) -> Self {
    self.seed_fn = Some(Box::new(seed_fn));
    self
  }

  /// The dimension $m$ of the domain $Omega$.
  pub fn dim(&self) -> Dim {
    self.dim
  }

  /// The fixed fallback seed for the chart's Gauss-Newton solve.
  pub fn seed(&self) -> &Coords<S> {
    &self.seed
  }

  /// The Gauss-Newton seed for the ambient point `p`: the seed heuristic if one
  /// was supplied, the fixed fallback otherwise.
  pub fn seed_at(&self, p: &Coord) -> Coords<S> {
    match &self.seed_fn {
      Some(heuristic) => heuristic(p),
      None => self.seed.clone(),
    }
  }

  /// $phi(u)$: the forward map into the ambient space $RR^N$.
  pub fn forward(&self, u: &Coords<S>) -> Coord {
    (self.forward)(u)
  }

  /// $dif phi(u)$: the forward Jacobian, an $N times m$ matrix. Exact if one was
  /// supplied, central finite difference otherwise.
  pub fn jacobian(&self, u: &Coords<S>) -> Matrix {
    match &self.jacobian {
      Some(exact) => exact(u),
      None => self.finite_diff_jacobian(u),
    }
  }

  fn finite_diff_jacobian(&self, u: &Coords<S>) -> Matrix {
    let ambient = self.forward(u).dim();
    let mut jac = Matrix::zeros(ambient, self.dim.index());
    let mut plus = u.clone();
    let mut minus = u.clone();
    for j in 0..self.dim.index() {
      plus.vector_mut()[j] += FD_STEP;
      minus.vector_mut()[j] -= FD_STEP;
      let column = (self.forward(&plus).vector() - self.forward(&minus).vector()) / (2.0 * FD_STEP);
      jac.set_column(j, &column);
      plus.vector_mut()[j] = u[j];
      minus.vector_mut()[j] = u[j];
    }
    jac
  }

  /// $chi(p) = phi^(-1)(r(p))$: the point of $Omega$ whose image is nearest the
  /// ambient point `p`, found from `seed`. Exact if a closed-form chart was
  /// supplied, Gauss-Newton on $norm(phi(u) - p)^2$ otherwise.
  pub fn chart(&self, p: &Coord, seed: &Coords<S>) -> Coords<S> {
    match &self.chart {
      Some(exact) => exact(p, seed),
      None => self.gauss_newton(p, seed),
    }
  }

  fn gauss_newton(&self, p: &Coord, seed: &Coords<S>) -> Coords<S> {
    let mut u = seed.clone();
    for _ in 0..GN_MAX_ITER {
      let residual: Vector = self.forward(&u).vector() - p.vector();
      if residual.norm() < GN_TOL {
        break;
      }
      let step = self.chart_differential(&u) * residual;
      *u.vector_mut() -= step;
    }
    u
  }

  /// $dif chi(u) = (dif phi(u))^+$: the chart differential, an $m times N$
  /// matrix, the pseudo-inverse of the forward Jacobian.
  pub fn chart_differential(&self, u: &Coords<S>) -> Matrix {
    self
      .jacobian(u)
      .pseudo_inverse(PINV_TOL)
      .expect("forward Jacobian has no pseudo-inverse")
  }

  /// The metric $g = phi^* delta = (dif phi)^T dif phi$ that $phi$ induces on
  /// the domain $Omega$ at `u`: the pullback of the ambient Euclidean metric,
  /// the Gramian of the tangent vectors $diff_i phi$.
  ///
  /// This is the distortion of the parametrization made explicit, and the datum
  /// the continuum unlocks downstream: a parametrization-induced cell metric
  /// (sampling $g$ at the barycenter, closer to $g_M$ than the chord metric),
  /// and metric-aware meshing (placing cells so they are well-shaped in $g$, not
  /// in the flat coordinates). It presupposes no inverse and no closed form --
  /// only the Jacobian, which is always available.
  pub fn induced_metric(&self, u: &Coords<S>) -> Metric {
    Metric::new(Gramian::from_euclidean_vectors(self.jacobian(u)))
  }
}

impl Parametrization<Ambient> {
  /// The flat continuum: $Omega = RR^N$, $phi = id$. Its chart is the identity
  /// and its Jacobian the identity matrix, so no finite difference and no
  /// Gauss-Newton run. This is the value that makes `pullback_on` the identity
  /// special case of `pullback_through`.
  pub fn identity(dim: Dim) -> Self {
    Self::new(|u: &Coord| u.clone(), dim)
      .with_jacobian(move |_| Matrix::identity(dim.index(), dim.index()))
      .with_chart(|p: &Coord, _| p.clone())
  }

  /// The $n$-sphere $S^n subset RR^(n+1)$ of the given `radius`, in
  /// hyperspherical coordinates, dimension-general: `dim` $= n$ is the intrinsic
  /// dimension, so $S^1$ is the circle, $S^2$ the ordinary sphere, and the
  /// recursion continues.
  ///
  /// The domain is the angle box $Omega = \[0, pi\]^(n-1) times \[0, 2 pi)$ and the
  /// ambient space is $RR^(n+1)$. The forward map is
  ///
  /// $ x_1 = r cos phi_1, quad x_k = r (product_(j<k) sin phi_j) cos phi_k, quad
  ///   x_(n+1) = r product_(j=1)^n sin phi_j. $
  ///
  /// The chart is its closed form. The angles are scale-invariant, so the same
  /// inverse serves every radius and *is* the radial nearest-point projection
  /// onto the sphere: the orthogonal retraction, with no Gauss-Newton. The
  /// Jacobian is left to the finite difference.
  pub fn sphere(dim: Dim, radius: f64) -> Self {
    assert!(dim >= 1, "the 0-sphere has no chart");
    Self::new(
      move |angle: &Coord| Coord::new(hyperspherical(radius, angle)),
      dim,
    )
    .with_chart(move |p: &Coord, _| Coord::new(hyperspherical_angles(p)))
  }

  /// The graph of a height function $h: Omega -> RR$ over an $n$-dimensional
  /// domain, as the surface $u |-> (u, h(u)) subset RR^(n+1)$. Dimension-general.
  ///
  /// A graph is an immersion with a full-rank Jacobian *everywhere* -- no
  /// coordinate singularity -- so unlike the sphere it needs no closed-form
  /// chart: the derived path (finite-difference Jacobian, Gauss-Newton) is
  /// robust. The seed heuristic is the vertical drop $p |-> p_(1..n)$, which is
  /// $O(norm(dif h))$ from the true footpoint and keeps the solve in-basin from
  /// any ambient point.
  pub fn graph(height: impl Fn(&Coord) -> f64 + Sync + 'static, dim: Dim) -> Self {
    Self::new(
      move |u: &Coord| Coord::new(u.vector().clone().insert_row(dim.index(), height(u))),
      dim,
    )
    .with_seed_fn(move |p: &Coord| Coord::new(p.rows(0, dim.index()).into_owned()))
  }

  /// The solid $n$-ball in spherical coordinates
  /// $(r, phi_1, dots, phi_(n-1)) |-> RR^n$, dimension-general: the disk in polar
  /// coordinates at $n = 2$, the ball in spherical coordinates at $n = 3$.
  ///
  /// A flat region of $RR^n$ written in a *curvilinear* chart: the metric is
  /// Euclidean and a mesh of it is exact, so pulling a form stated in these
  /// coordinates isolates the curvilinear-chart Jacobian from any $M_h != M$
  /// domain gap. Its boundary is a [`Self::sphere`]. The chart is closed form,
  /// from the same hyperspherical inverse.
  ///
  /// The radial extent is not a datum here: the forward map reads $r$ from the
  /// coordinate, and a [`Parametrization`] carries no domain bounds. The ball's
  /// radius is set by wherever `r` ranges on the mesh, not by this constructor.
  pub fn ball(dim: Dim) -> Self {
    assert!(
      dim >= 2,
      "the 1-ball is an interval, not a curvilinear chart"
    );
    Self::new(
      move |u: &Coord| {
        let r = u[0];
        let angles = u.rows(1, dim.index() - 1).into_owned();
        Coord::new(hyperspherical(r, &Coord::new(angles)))
      },
      dim,
    )
    .with_chart(move |p: &Coord, _| {
      let mut u = Vector::zeros(dim.index());
      u[0] = p.norm();
      u.rows_mut(1, dim.index() - 1)
        .copy_from(&hyperspherical_angles(p));
      Coord::new(u)
    })
  }

  /// The 2-torus of revolution in $RR^3$, tube radius `minor` swept at distance
  /// `major` from the axis: $(theta, phi) |-> ((R + r cos theta) cos phi,
  /// (R + r cos theta) sin phi, r sin theta)$.
  ///
  /// The first curved geometry past the sphere with *varying* Gaussian curvature
  /// (positive on the outer rim, negative on the inner) and *nontrivial*
  /// cohomology, $dim H^1 = 2$: it carries genuine harmonic 1-forms, which is
  /// what exercises the harmonic-projection path a simply connected sphere leaves
  /// untouched. The chart is the closed-form toroidal-angle inverse.
  pub fn torus(major: f64, minor: f64) -> Self {
    assert!(major > minor && minor > 0.0, "not an embedded torus");
    Self::new(
      move |u: &Coord| {
        let (theta, phi) = (u[0], u[1]);
        let rho = major + minor * theta.cos();
        Coord::from_iterator(3, [rho * phi.cos(), rho * phi.sin(), minor * theta.sin()])
      },
      Dim::new(2),
    )
    .with_chart(move |p: &Coord, _| {
      let phi = p[1].atan2(p[0]);
      let rho = (p[0] * p[0] + p[1] * p[1]).sqrt();
      let theta = p[2].atan2(rho - major);
      Coord::from_iterator(2, [theta, phi])
    })
  }
}

/// The hyperspherical forward map: $n$ angles to a point of radius `radius` in
/// $RR^(n+1)$. Shared by the sphere and the ball.
fn hyperspherical(radius: f64, angles: &Coord) -> Vector {
  let dim = angles.dim();
  let mut x = Vector::zeros(dim + 1);
  let mut sin_prod = radius;
  for k in 0..dim {
    x[k] = sin_prod * angles[k].cos();
    sin_prod *= angles[k].sin();
  }
  x[dim] = sin_prod;
  x
}

/// The hyperspherical inverse: the $n$ angles of a point in $RR^(n+1)$,
/// scale-invariant and hence radius-free.
fn hyperspherical_angles(p: &Coord) -> Vector {
  let dim = p.dim() - 1;
  let mut phi = Vector::zeros(dim);
  for k in 0..dim - 1 {
    let tail = p.iter().skip(k + 1).map(|v| v * v).sum::<f64>().sqrt();
    phi[k] = tail.atan2(p[k]);
  }
  phi[dim - 1] = p[dim].atan2(p[dim - 1]);
  phi
}

#[cfg(test)]
mod test {
  use super::*;

  use approx::assert_relative_eq;

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

  /// The Gauss-Newton chart lands *on* the manifold when the target is off it:
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
      assert_relative_eq!(g.vector_gramian().matrix(), &expected, epsilon = 1e-6);
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
}
