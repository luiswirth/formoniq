use crate::galerkin::{BilinearForm, LinearForm};

use {
  derham::{
    interpolate::{form::WhitneyExpansion, samples::LsfSamples},
    section::Section,
  },
  metric::{
    Metric,
    tensor::{TensorExt, inner, multiform_metric},
  },
  multialgebra::{Dim, ExteriorGrade, Tensor, exterior_power},
  multiindex::{Combination, Sign},
  regge::cell_volume,
  simplicial::{
    atlas::{
      Bary, Chart, ChartExt, FaceTrace, MeshPoint, SimplexQuadRule, face_bary_to_cell_bary,
      unit_bary_gramian, unit_difbarys, unit_simplex_volume,
    },
    linalg::{Matrix, Vector},
    topology::simplex::unit_boundary_operator,
  },
};

/// The scalar mass form under the trapezoidal rule, whose element matrix is
/// diagonal: the lumped mass.
pub struct ScalarLumpedMass;
impl BilinearForm for ScalarLumpedMass {
  fn test_grade(&self) -> ExteriorGrade {
    Dim::ZERO
  }
  fn trial_grade(&self) -> ExteriorGrade {
    Dim::ZERO
  }
  fn element(&self, metric: &Metric, _chart: Chart) -> Matrix {
    let n = metric.dim() + 1;
    let v = cell_volume(metric) / n as f64;
    Matrix::from_diagonal_element(n, n, v)
  }
}

/// The grade-$k$ Whitney mass matrix, the weak Hodge star,
///
/// $M = [inner(star lambda_tau, lambda_sigma)_(L^2 Lambda^k (K))]_(sigma,tau in Delta_k (K))$.
///
/// The kernel every [`WhitneyPairing`] sandwiches, and reached as an element
/// matrix through [`WhitneyPairing::mass`] rather than directly: the mass is
/// the pairing whose two sides are both undifferentiated, not a separate
/// operator.
///
/// The integrand splits into a blade half and a polynomial half, so
/// $M = vol_K C^top (H times.o Q) C$, with $H = D (Lambda^k g^(-1)) D^top$
/// the Gramian of the barycentric $k$-blades $dif lambda_I$,
/// $Q_(v w) = (1 + delta_(v w)) \/ ((n+1)(n+2))$ the unit-volume scalar mass,
/// and $C$ the coefficient map of the deletion formula
/// $W_sigma = k! sum_i (-1)^i lambda_(sigma_i) dif lambda_(sigma without sigma_i)$.
/// Only $H$ depends on the metric, so $M$ is linear in $Lambda^k g^(-1)$.
pub struct HodgeMass {
  dim: Dim,
  grade: ExteriorGrade,
  /// $C$, the Whitney basis as a map into blades times coordinates.
  expansion: WhitneyExpansion,
  /// $Lambda^k$ of the reference barycentric differentials: the pullback
  /// matrix taking formal barycentric $k$-blades to reference $k$-forms.
  difbarys_power: Matrix,
  /// $Q$, the barycentric half, at unit volume.
  bary_gramian: Matrix,
}
impl HodgeMass {
  pub fn new(dim: impl Into<Dim>, grade: impl Into<ExteriorGrade>) -> Self {
    let (dim, grade) = (dim.into(), grade.into());
    Self {
      dim,
      grade,
      expansion: WhitneyExpansion::new(dim, grade),
      difbarys_power: exterior_power(&unit_difbarys(dim), grade),
      bary_gramian: unit_bary_gramian(dim),
    }
  }

  /// Chart-free: every chart of the atlas is the reference cell up to the
  /// labelling of its vertices, so the geometry enters through the metric
  /// alone.
  pub fn element(&self, metric: &Metric) -> Matrix {
    assert_eq!(self.dim, metric.dim());

    // $H$: the Gramian of the barycentric $k$-blades $lambda^* (e_I)$, one
    // Cauchy-Binet sandwich for all Whitney wedge terms at once.
    let form_gramian = multiform_metric(metric, self.grade);
    let blade_gramian =
      &self.difbarys_power * form_gramian.matrix() * self.difbarys_power.transpose();

    cell_volume(metric) * self.expansion.pullback(&blade_gramian, &self.bary_gramian)
  }
}

/// Which family a side of a [`WhitneyPairing`] ranges over: the grade-$k$
/// shape functions themselves, or the exterior derivatives of those one grade
/// below.
///
/// [`Difs`](Self::Difs) is metric-free (the exterior derivative of a Whitney
/// form is the coboundary of the reference cell, a $plus.minus 1$ incidence),
/// which is why the geometry of a pairing enters through its mass alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WhitneyFamily {
  /// $lambda_sigma$, $sigma in Delta_k (K)$.
  Forms,
  /// $dif lambda_sigma$, $sigma in Delta_(k-1) (K)$.
  Difs,
}
impl WhitneyFamily {
  /// How far below the pairing's grade this family's degrees of freedom sit.
  fn grade_offset(self) -> ExteriorGrade {
    ExteriorGrade::from(usize::from(self == Self::Difs))
  }
}

/// Element matrix of a bilinear form pairing two Whitney families in the
/// grade-$k$ inner product,
///
/// $A = R^top M_k C$,
///
/// with $C$ the trial side and $R$ the test side, each either the shape
/// functions of grade $k$ or, one grade below, their exterior derivative
/// $D$, the coboundary of the reference cell; $R^top = D^top = partial$ is then
/// its boundary. The four choices are the four blocks a problem posed around
/// grade $k$ is built from, and they are named by the differentiated side:
/// [`mass`](Self::mass), [`dif_trial`](Self::dif_trial),
/// [`dif_test`](Self::dif_test) and [`dif_both`](Self::dif_both).
///
/// The grade argument is the grade of the *inner product* in all four, so it
/// is the grade of $M_k$ and never that of the degrees of freedom, which sit
/// one below on a differentiated side.
pub struct WhitneyPairing {
  mass: HodgeMass,
  test: WhitneyFamily,
  trial: WhitneyFamily,
  /// $partial$, applied on the left, where the rows are one grade below the mass.
  row: Option<Matrix>,
  /// $D$, applied on the right, where the columns are.
  col: Option<Matrix>,
}

impl WhitneyPairing {
  /// The pairing of `test` against `trial` in the grade-`grade` inner product.
  ///
  /// Private, with the four combinations as the constructors: which family a
  /// side ranges over is not a free parameter a caller has an opinion about,
  /// it is which of the four blocks is wanted.
  fn new(
    dim: impl Into<Dim>,
    grade: impl Into<ExteriorGrade>,
    test: WhitneyFamily,
    trial: WhitneyFamily,
  ) -> Self {
    let (dim, grade) = (dim.into(), grade.into());
    let boundary = || unit_boundary_operator(dim, grade);
    Self {
      mass: HodgeMass::new(dim, grade),
      test,
      trial,
      row: (test == WhitneyFamily::Difs).then(boundary),
      col: (trial == WhitneyFamily::Difs).then(|| boundary().transpose()),
    }
  }

  /// The mass form $(u, v)$,
  /// $A = [inner(lambda_J, lambda_I)_(L^2 Lambda^k (K))]_(I,J in Delta_k (K))$.
  pub fn mass(dim: impl Into<Dim>, grade: impl Into<ExteriorGrade>) -> Self {
    Self::new(dim, grade, WhitneyFamily::Forms, WhitneyFamily::Forms)
  }

  /// The weak mixed exterior derivative $(dif sigma, v)$,
  /// $A = [inner(dif lambda_J, lambda_I)_(L^2 Lambda^k (K))]_(I in Delta_k, J in Delta_(k-1) (K))$.
  pub fn dif_trial(dim: impl Into<Dim>, grade: impl Into<ExteriorGrade>) -> Self {
    Self::new(dim, grade, WhitneyFamily::Forms, WhitneyFamily::Difs)
  }

  /// The weak mixed codifferential $(u, dif tau)$,
  /// $A = [inner(lambda_J, dif lambda_I)_(L^2 Lambda^k (K))]_(I in Delta_(k-1), J in Delta_k (K))$,
  /// the transpose of [`dif_trial`](Self::dif_trial) at the same grade.
  pub fn dif_test(dim: impl Into<Dim>, grade: impl Into<ExteriorGrade>) -> Self {
    Self::new(dim, grade, WhitneyFamily::Difs, WhitneyFamily::Forms)
  }

  /// The stiffness form $(dif u, dif v)$,
  /// $A = [inner(dif lambda_J, dif lambda_I)_(L^2 Lambda^k (K))]_(I,J in Delta_(k-1) (K))$,
  /// whose degrees of freedom therefore sit at grade $k-1$.
  pub fn dif_both(dim: impl Into<Dim>, grade: impl Into<ExteriorGrade>) -> Self {
    Self::new(dim, grade, WhitneyFamily::Difs, WhitneyFamily::Difs)
  }
}

impl BilinearForm for WhitneyPairing {
  fn test_grade(&self) -> ExteriorGrade {
    self.mass.grade - self.test.grade_offset()
  }
  fn trial_grade(&self) -> ExteriorGrade {
    self.mass.grade - self.trial.grade_offset()
  }
  fn element(&self, metric: &Metric, _chart: Chart) -> Matrix {
    let mass = self.mass.element(metric);
    let mass = match &self.col {
      Some(dif) => mass * dif,
      None => mass,
    };
    match &self.row {
      Some(codif) => codif * mass,
      None => mass,
    }
  }
}

/// An element integral over a cell: a quadrature rule and the mesh points its
/// nodes sit at.
///
/// The shape functions are not held here. They arrive as [`LsfSamples`] built
/// against this rule's [`nodes`](Self::nodes), so one routine serves the
/// Whitney basis, its differentials, or two grades at once. What stays per-cell
/// is the chart, the metric and the volume, a coefficient being a [`Section`]
/// evaluated at the [`MeshPoint`]s.
pub struct CellQuadrature {
  qr: SimplexQuadRule,
  nodes: Vec<Bary>,
}
impl CellQuadrature {
  /// `qr` defaults to the degree-1 Grundmann-Möller rule, the cheapest rule
  /// that is exact on affine integrands.
  pub fn new(dim: impl Into<Dim>, qr: Option<SimplexQuadRule>) -> Self {
    let dim = dim.into();
    let qr = qr.unwrap_or(SimplexQuadRule::degree(dim, 1));
    let nodes = qr.points().map(|bary| bary.to_coords()).collect();
    Self { qr, nodes }
  }

  /// The nodes in barycentric coordinates: what an [`LsfSamples`] table is
  /// built against.
  pub fn nodes(&self) -> &[Bary] {
    &self.nodes
  }

  fn point(&self, chart: Chart, inode: usize) -> MeshPoint {
    chart.point(self.nodes[inode].clone())
  }

  /// $[integral_K f(x, W_sigma (x)) vol]_sigma$.
  pub fn integrate<F>(&self, shapes: &LsfSamples, chart: Chart, vol: f64, f: F) -> Vector
  where
    F: Fn(&MeshPoint, &Tensor) -> f64,
  {
    assert_eq!(shapes.nnodes(), self.nodes.len());

    let mut elvec = Vector::zeros(shapes.ndofs());
    for (inode, weight) in self.qr.weights().iter().enumerate() {
      let point = self.point(chart, inode);
      for (i, value) in shapes.at_node(inode).iter().enumerate() {
        elvec[i] += weight * f(&point, value);
      }
    }
    vol * elvec
  }

  /// $[integral_K f(x, W_sigma (x), W'_tau (x)) vol]_(sigma tau)$, the two
  /// families being free to sit at different grades, which is what a mixed
  /// block needs.
  pub fn integrate_pair<F>(
    &self,
    rows: &LsfSamples,
    cols: &LsfSamples,
    chart: Chart,
    vol: f64,
    f: F,
  ) -> Matrix
  where
    F: Fn(&MeshPoint, &Tensor, &Tensor) -> f64,
  {
    assert_eq!(rows.nnodes(), self.nodes.len());
    assert_eq!(cols.nnodes(), self.nodes.len());

    let mut elmat = Matrix::zeros(rows.ndofs(), cols.ndofs());
    for (inode, weight) in self.qr.weights().iter().enumerate() {
      let point = self.point(chart, inode);
      for (i, row) in rows.at_node(inode).iter().enumerate() {
        for (j, col) in cols.at_node(inode).iter().enumerate() {
          elmat[(i, j)] += weight * f(&point, row, col);
        }
      }
    }
    vol * elmat
  }
}

/// A facet of the reference cell, as $partial K$ presents it.
struct BoundaryFacet {
  /// The sign the boundary operator induces, $(-1)^i$ for the facet omitting
  /// the $i$-th vertex.
  sign: f64,
  /// The facet's local vertex positions within the cell.
  positions: Combination,
  /// The trace onto this facet at its own top grade $n-1$, where the
  /// $(n-1)$-form integrand becomes a scalar.
  trace: FaceTrace,
}

/// Quadrature over $partial K$ for an element integral: the cell's facets, each
/// integrated in the cell's chart and weighted by the sign the boundary
/// operator induces.
///
/// Metric-free, because the integrand is an $(n-1)$-form rather than a
/// scalar against $vol$: a form over a simplex of its own grade carries its own
/// geometry. An integrand reads whatever metric it wants for itself.
///
/// The quadrature applies the [`FaceTrace`] onto each facet, so a caller cannot
/// forget that only the tangential part of a form is integrable over a face.
/// The facets are the cell's own and carry the cell's own DOFs, so the result
/// is an element matrix and ordinary assembly scatters it.
pub struct BoundaryQuadrature {
  dim: Dim,
  /// Facet-major: node `f * npoints + q` lies on facet `f`.
  nodes: Vec<Bary>,
  weights: Vec<f64>,
  npoints: usize,
  facets: Vec<BoundaryFacet>,
}

impl BoundaryQuadrature {
  pub fn new(dim: impl Into<Dim>, qr: Option<SimplexQuadRule>) -> Self {
    let dim = dim.into();
    let facet_dim = dim - 1;
    let qr = qr.unwrap_or(SimplexQuadRule::degree(facet_dim, 1));

    let facets: Vec<_> = Combination::full((dim + 1).index())
      .deletions()
      .map(|(sign, _, positions)| BoundaryFacet {
        sign: sign.as_f64(),
        positions,
        trace: FaceTrace::new(dim, &positions, facet_dim),
      })
      .collect();

    // The facets' nodes, scattered into the cell's barycentric coordinates so
    // that one shape-function table covers the whole boundary.
    let nodes = facets
      .iter()
      .flat_map(|facet| {
        qr.points()
          .map(|bary| face_bary_to_cell_bary(dim, &facet.positions, bary))
          .collect::<Vec<_>>()
      })
      .collect();
    let weights = facets
      .iter()
      .flat_map(|_| qr.weights().iter().copied().collect::<Vec<_>>())
      .collect();

    Self {
      dim,
      nodes,
      weights,
      npoints: qr.npoints(),
      facets,
    }
  }

  /// The nodes of the whole boundary, in the cell's barycentric coordinates:
  /// what an [`LsfSamples`] table is built against.
  pub fn nodes(&self) -> &[Bary] {
    &self.nodes
  }

  /// $integral_(partial K) omega$ of a section of grade $n-1$.
  ///
  /// A field, not a closure: whether it is analytic data pulled back from a
  /// continuum, the interpolation of a cochain, or a combinator over either is
  /// invisible here, which is what makes natural boundary data intrinsic by the
  /// same code path that serves an embedded source.
  pub fn integrate_form(&self, chart: Chart, form: &impl Section) -> f64 {
    assert_eq!(form.dim(), self.dim);
    assert_eq!(
      form.grade(),
      self.dim - 1,
      "A boundary integrand is a form of grade n-1."
    );

    let mut integral = 0.0;
    for (inode, bary) in self.nodes.iter().enumerate() {
      let facet = &self.facets[inode / self.npoints];
      let point = chart.point(bary.clone());
      integral += facet.sign * self.weights[inode] * facet.trace.top_coefficient(&form.at(&point));
    }
    unit_simplex_volume(self.dim - 1) * integral
  }

  /// $[integral_(partial K) f(x, W_sigma, W'_tau)]_(sigma tau)$, where `f` is the
  /// pointwise integrand of the bilinear form: at each point a bilinear map
  /// $Lambda^(k_r) times Lambda^(k_c) -> Lambda^(n-1)$, hence a section of
  /// $"Hom"(Lambda^(k_r) times.o Lambda^(k_c), Lambda^(n-1))$ evaluated
  /// against the two shape functions.
  ///
  /// It is a family indexed by pairs of degrees of freedom, so it cannot be one
  /// [`Section`]. That is the whole difference between this and
  /// [`Self::integrate_form`].
  pub fn integrate_pair<F>(
    &self,
    rows: &LsfSamples,
    cols: &LsfSamples,
    chart: Chart,
    f: F,
  ) -> Matrix
  where
    F: Fn(&MeshPoint, &Tensor, &Tensor) -> Tensor,
  {
    assert_eq!(rows.nnodes(), self.nodes.len());
    assert_eq!(cols.nnodes(), self.nodes.len());

    let mut elmat = Matrix::zeros(rows.ndofs(), cols.ndofs());
    for (inode, bary) in self.nodes.iter().enumerate() {
      let facet = &self.facets[inode / self.npoints];
      let point = chart.point(bary.clone());
      let weight = facet.sign * self.weights[inode];

      for (i, row) in rows.at_node(inode).iter().enumerate() {
        for (j, col) in cols.at_node(inode).iter().enumerate() {
          elmat[(i, j)] += weight * facet.trace.top_coefficient(&f(&point, row, col));
        }
      }
    }
    unit_simplex_volume(self.dim - 1) * elmat
  }
}

/// Element matrix of the Hodge mass bilinear form weighted by a scalar
/// coefficient field,
/// $[integral_K alpha inner(W_sigma, W_tau)_(Lambda^k) vol]_(sigma tau)$.
///
/// The varying-coefficient counterpart of [`HodgeMass`], which is exact
/// where this is a quadrature: with $alpha equiv 1$ the two agree to the
/// accuracy of the rule. Intrinsic, like every element integral here, the
/// coefficient is a grade-0 section of the manifold, so a metric never enters
/// through it, only through the inner product on $Lambda^k$.
pub struct WeightedHodgeMass<'a, F> {
  coefficient: &'a F,
  grade: ExteriorGrade,
  quad: CellQuadrature,
  shapes: LsfSamples,
}
impl<'a, F: Section> WeightedHodgeMass<'a, F> {
  /// Panics unless the coefficient is a grade-0 section: a weight is a scalar.
  pub fn new(
    coefficient: &'a F,
    grade: impl Into<ExteriorGrade>,
    qr: Option<SimplexQuadRule>,
  ) -> Self {
    assert_eq!(
      coefficient.grade(),
      Dim::ZERO,
      "A scalar coefficient must be a grade-0 section."
    );
    let grade = grade.into();
    let quad = CellQuadrature::new(coefficient.dim(), qr);
    let shapes = LsfSamples::whitney(coefficient.dim(), grade, quad.nodes());
    Self {
      coefficient,
      grade,
      quad,
      shapes,
    }
  }
}
impl<F: Sync + Section> BilinearForm for WeightedHodgeMass<'_, F> {
  fn test_grade(&self) -> ExteriorGrade {
    self.grade
  }
  fn trial_grade(&self) -> ExteriorGrade {
    self.grade
  }
  fn element(&self, metric: &Metric, chart: Chart) -> Matrix {
    self.quad.integrate_pair(
      &self.shapes,
      &self.shapes,
      chart,
      cell_volume(metric),
      |point, row, col| self.coefficient.at(point).as_scalar() * inner(row, col, metric),
    )
  }
}

/// Element matrix of the weak Lie derivative $cal(L)_v$ along a prescribed
/// vector field, at any grade,
///
/// $$ a_K (omega, eta) = integral_K inner(iota_v dif omega, eta) vol
///    + integral_(diff K) (iota_v omega) wedge star eta. $$
///
/// Cartan's $cal(L)_v = iota_v dif + dif iota_v$ gives the two terms. The
/// second is a boundary integral because the shape functions are coclosed on a
/// cell, so integrating it by parts leaves nothing in the interior. The two
/// terms are the two degenerate grades, and so cover the classical pair:
/// advective form at $k = 0$, conservation form at $k = n$.
///
/// The boundary term's star is taken in the cell's reference frame, and needs
/// no coherent orientation: flipping that frame flips both the star and the
/// induced orientation of $partial K$, and the product is what the term is. So
/// assembly stays independent of a gauge it must not depend on, and the
/// operator exists on a non-orientable mesh.
///
/// `velocity` is a vector field, not a 1-form, and nothing is sharped here:
/// $iota_v$ and $dif$ are metric-free, and the metric enters only through the
/// $L^2$ pairing and the star.
///
/// Central and unstabilized: each cell integrates its own trace of a shared
/// facet, so no numerical flux is chosen. Conservative at both ends of the
/// grade range, where the defect $integral_(partial K) inner(omega, eta) iota_v
/// vol$ vanishes: the shape functions are continuous at $k = 0$ and constant
/// per cell at $k = n$. Dispersive throughout, it damps nothing, so the phase
/// error of barely resolved modes persists as oscillation, which conservation
/// does not see.
pub struct LieDerivative<'a, V> {
  velocity: &'a V,
  grade: ExteriorGrade,
  volume: CellQuadrature,
  boundary: BoundaryQuadrature,
  /// $W_sigma$ at the volume nodes: the test functions.
  test: LsfSamples,
  /// $dif W_tau$ at the volume nodes, of grade $k+1$.
  trial_dif: LsfSamples,
  /// $W_sigma$ and $W_tau$ at the boundary nodes.
  boundary_test: LsfSamples,
  boundary_trial: LsfSamples,
}

impl<'a, V: Section> LieDerivative<'a, V> {
  /// One `quad_degree` serves both integrals, at their own dimensions.
  ///
  /// Degree $2 + p$ is exact for a velocity of polynomial degree $p$: the
  /// interior integrand pairs a constant $dif W$ against an affine $W$, the
  /// boundary one two affine shape functions, so the boundary is the binding
  /// side and $2$ suffices for a constant velocity.
  ///
  /// Panics unless the velocity is a grade-1 section: a vector field.
  pub fn new(velocity: &'a V, grade: impl Into<ExteriorGrade>, quad_degree: usize) -> Self {
    assert_eq!(
      velocity.grade(),
      Dim::ONE,
      "A velocity is a grade-1 section, a vector field."
    );
    let (dim, grade) = (velocity.dim(), grade.into());

    let volume = CellQuadrature::new(dim, Some(SimplexQuadRule::degree(dim, quad_degree)));
    let boundary =
      BoundaryQuadrature::new(dim, Some(SimplexQuadRule::degree(dim - 1, quad_degree)));
    Self {
      velocity,
      grade,
      test: LsfSamples::whitney(dim, grade, volume.nodes()),
      trial_dif: LsfSamples::whitney_dif(dim, grade, volume.nodes().len()),
      boundary_test: LsfSamples::whitney(dim, grade, boundary.nodes()),
      boundary_trial: LsfSamples::whitney(dim, grade, boundary.nodes()),
      volume,
      boundary,
    }
  }
}

impl<V: Sync + Section> BilinearForm for LieDerivative<'_, V> {
  fn test_grade(&self) -> ExteriorGrade {
    self.grade
  }
  fn trial_grade(&self) -> ExteriorGrade {
    self.grade
  }

  fn element(&self, metric: &Metric, chart: Chart) -> Matrix {
    let interior = self.volume.integrate_pair(
      &self.test,
      &self.trial_dif,
      chart,
      cell_volume(metric),
      |point, test, trial_dif| {
        let advected = trial_dif.interior_product(&self.velocity.at(point));
        inner(&advected, test, metric)
      },
    );

    let boundary = self.boundary.integrate_pair(
      &self.boundary_test,
      &self.boundary_trial,
      chart,
      |point, test, trial| {
        trial
          .interior_product(&self.velocity.at(point))
          .wedge(&test.star(metric, Sign::Pos))
      },
    );

    interior + boundary
  }
}

/// Element vector of the source load
/// $[integral_K inner(f, W_sigma)_(Lambda^k) vol]_(sigma in Delta_k (K))$.
///
/// Intrinsic: the source is a field on the manifold, the Whitney shape
/// functions are the reference ones, and both are paired in the cell's
/// reference frame under the induced inner product $Lambda^k g^(-1)$ of the
/// cell metric. Source assembly therefore runs on Regge geometry, with no
/// coordinates in sight.
pub struct SourceForm<'a, F> {
  source: &'a F,
  quad: CellQuadrature,
  shapes: LsfSamples,
}
impl<'a, F: Section> SourceForm<'a, F> {
  pub fn new(source: &'a F, qr: Option<SimplexQuadRule>) -> Self {
    let quad = CellQuadrature::new(source.dim(), qr);
    let shapes = LsfSamples::whitney(source.dim(), source.grade(), quad.nodes());
    Self {
      source,
      quad,
      shapes,
    }
  }
}
impl<F: Sync + Section> LinearForm for SourceForm<'_, F> {
  fn test_grade(&self) -> ExteriorGrade {
    self.source.grade()
  }
  fn element(&self, metric: &Metric, chart: Chart) -> Vector {
    self.quad.integrate(
      &self.shapes,
      chart,
      cell_volume(metric),
      |point, whitney| inner(&self.source.at(point), whitney, metric),
    )
  }
}
