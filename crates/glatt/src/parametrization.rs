//! The smooth parametrization of the continuum, and the chart it induces.
//!
//! The one required datum of a continuum $M$ is its forward map
//! $phi: Omega -> RR^N$, from a coordinate domain out into the ambient space,
//! a parametrization, pointing into the manifold, exactly opposite to a chart.
//! Everything else is derived, because requiring the inverse would ask for data
//! the mathematics already fixes:
//!
//! - the differential $dif phi$ (an $N times m$ matrix) by central finite
//!   difference of $phi$, unless an exact Jacobian is supplied;
//! - the chart $chi = phi^(-1) compose r$ by Gauss-Newton on
//!   $min_u norm(phi(u) - p)_2^2$, unless a closed form is supplied.
//!
//! Least-squares in the ambient Euclidean norm is the orthogonal (nearest
//! point) projection onto $op("im") phi$, so the derived retraction $r$ is not
//! a retraction but the geometrically optimal one, which is what makes the
//! domain gap $O(h^2)$ rather than $O(h)$ (Dziuk, Demlow). The chart
//! differential $dif chi = (dif phi)^+$ is the pseudo-inverse of the forward
//! Jacobian: a genuine left inverse because $phi$ is an immersion.
//!
//! Metric-independent by construction: the pseudo-inverse here is the
//! Moore-Penrose one for the ambient Euclidean metric, which is what "nearest
//! point" means, and no `Metric` of the continuum enters.

use coorder::{Ambient, Coord, CoordSpace, Coords, Matrix, Vector};
use metric::Metric;
use multialgebra::Dim;

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
/// The convergence tolerance and iteration cap for the Gauss-Newton chart. The
/// tolerance is on the step, relative to the domain point it moves, so it means
/// the same thing on a domain of any scale.
const GN_TOL: f64 = 1e-12;
pub const GN_MAX_ITER: usize = 100;
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

  /// Supply a seed heuristic $p |-> u_0$ that guesses a domain point from an
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
    let mut previous = f64::INFINITY;
    for _ in 0..GN_MAX_ITER {
      // What vanishes at the nearest point is the step, not the residual: the
      // residual there is the distance from `p` to the image, and it is zero
      // only for a `p` already on the manifold. The step is that residual
      // projected onto the tangent space, hence the first-order optimality
      // condition of the least-squares problem this solves.
      let residual: Vector = &self.forward(&u) - p;
      let step = self.chart_differential(&u) * residual;
      let size = step.norm();
      u -= &step;

      // Converged, or making no further progress. A derived Jacobian is only
      // accurate to the noise of its finite differences, and the step cannot
      // shrink past what that noise contributes, so stagnation is the floor an
      // approximate model imposes on any solve built from it.
      if size <= GN_TOL * (1.0 + u.vector().norm()) || size >= previous {
        break;
      }
      previous = size;
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
  /// the Gramian of the tangent vectors $partial_i phi$.
  ///
  /// This is the distortion of the parametrization made explicit, and the datum
  /// the continuum unlocks downstream: a parametrization-induced cell metric
  /// (sampling $g$ at the barycenter, closer to $g_M$ than the chord metric),
  /// and metric-aware meshing (placing cells so they are well-shaped in $g$, not
  /// in the flat coordinates). It presupposes no inverse and no closed form,
  /// only the Jacobian, which is always available.
  pub fn induced_metric(&self, u: &Coords<S>) -> Metric {
    Metric::from_euclidean_vectors(self.jacobian(u))
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
  /// inverse serves every radius and is the radial nearest-point projection
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
  /// A graph is an immersion with a full-rank Jacobian everywhere, no
  /// coordinate singularity, so unlike the sphere it needs no closed-form
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
  /// A flat region of $RR^n$ written in a curvilinear chart: the metric is
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
  /// The first curved geometry past the sphere with varying Gaussian curvature
  /// (positive on the outer rim, negative on the inner) and nontrivial
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
