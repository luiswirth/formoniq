use gramian::tensor::{TensorExt, multiform_gramian};
use {
  derham::{
    interpolate::{form::WhitneyExpansion, samples::LsfSamples},
    section::Section,
    trace::FaceTrace,
  },
  gramian::{Gramian, Metric},
  multialgebra::{Dim, ExteriorGrade, Tensor, exterior_power},
  multiindex::{Combination, Sign},
  simplicial::{
    atlas::{
      Bary, Chart, ChartExt, MeshPoint, SimplexQuadRule, face_bary_to_cell_bary, unit_bary_gramian,
      unit_difbarys, unit_simplex_volume,
    },
    geometry::cell_volume,
    linalg::{Matrix, Vector},
    topology::simplex::unit_boundary_operator,
  },
};

pub type ElMat = Matrix;
pub trait ElMatProvider: Sync {
  fn row_grade(&self) -> ExteriorGrade;
  fn col_grade(&self) -> ExteriorGrade;
  fn eval(&self, metric: &Metric, chart: Chart) -> ElMat;
}

/// Approximated Element Matrix Provider for scalar mass bilinear form,
/// obtained through trapezoidal quadrature rule.
pub struct ScalarLumpedMassElmat;
impl ElMatProvider for ScalarLumpedMassElmat {
  fn row_grade(&self) -> ExteriorGrade {
    Dim::ZERO
  }
  fn col_grade(&self) -> ExteriorGrade {
    Dim::ZERO
  }
  fn eval(&self, metric: &Metric, _chart: Chart) -> ElMat {
    let n = metric.dim() + 1;
    let v = cell_volume(metric) / n as f64;
    Matrix::from_diagonal_element(n, n, v)
  }
}

/// Element Matrix for the weak Hodge star operator / the mass bilinear form.
///
/// $M = [inner(star lambda_tau, lambda_sigma)_(L^2 Lambda^k (K))]_(sigma,tau in Delta_k (K))$
///
/// The integrand splits into a blade half and a polynomial half, so
/// $M = vol_K C^top (H times.circle Q) C$, with $H = D (Lambda^k g^(-1)) D^top$
/// the Gramian of the barycentric $k$-blades $dif lambda_I$,
/// $Q_(v w) = (1 + delta_(v w)) \/ ((n+1)(n+2))$ the unit-volume scalar mass,
/// and $C$ the coefficient map of the deletion formula
/// $W_sigma = k! sum_i (-1)^i lambda_(sigma_i) dif lambda_(sigma without sigma_i)$.
/// Only $H$ depends on the metric, so $M$ is linear in $Lambda^k g^(-1)$.
pub struct HodgeMassElmat {
  dim: Dim,
  grade: ExteriorGrade,
  /// $C$, the Whitney basis as a map into blades times coordinates.
  expansion: WhitneyExpansion,
  /// $Lambda^k$ of the reference barycentric differentials: the pullback
  /// matrix taking formal barycentric $k$-blades to reference $k$-forms.
  difbarys_power: Matrix,
  /// $Q$, the barycentric half, at unit volume.
  bary_gramian: Gramian,
}
impl HodgeMassElmat {
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
}
impl ElMatProvider for HodgeMassElmat {
  fn row_grade(&self) -> ExteriorGrade {
    self.grade
  }
  fn col_grade(&self) -> ExteriorGrade {
    self.grade
  }

  fn eval(&self, metric: &Metric, _chart: Chart) -> Matrix {
    assert_eq!(self.dim, metric.dim());

    // $H$: the Gramian of the barycentric $k$-blades $lambda^* (e_I)$, one
    // Cauchy-Binet sandwich for all Whitney wedge terms at once.
    let form_gramian = multiform_gramian(metric, self.grade);
    let blade_gramian =
      &self.difbarys_power * form_gramian.matrix() * self.difbarys_power.transpose();

    cell_volume(metric)
      * self
        .expansion
        .pullback(&blade_gramian, self.bary_gramian.matrix())
  }
}

/// Element Matrix Provider for the weak mixed exterior derivative $(dif sigma, v)$.
///
/// $A = [inner(dif lambda_J, lambda_I)_(L^2 Lambda^k (K))]_(I in Delta_, J in Delta_(k-1) (K))$
pub struct DifElmat {
  mass: HodgeMassElmat,
  dif: Matrix,
}
impl DifElmat {
  pub fn new(dim: impl Into<Dim>, grade: impl Into<ExteriorGrade>) -> Self {
    let (dim, grade) = (dim.into(), grade.into());
    let mass = HodgeMassElmat::new(dim, grade);
    let dif = unit_boundary_operator(dim, grade).transpose();
    Self { mass, dif }
  }
}

impl ElMatProvider for DifElmat {
  fn row_grade(&self) -> ExteriorGrade {
    self.mass.grade
  }
  fn col_grade(&self) -> ExteriorGrade {
    self.mass.grade - 1
  }
  fn eval(&self, metric: &Metric, chart: Chart) -> Matrix {
    let mass = self.mass.eval(metric, chart);
    mass * &self.dif
  }
}

/// Element Matrix Provider for the weak mixed codifferential $(u, dif tau)$.
///
/// $A = [inner(lambda_J, dif lambda_I)_(L^2 Lambda^k (K))]_(I in Delta_(k-1), J in Delta_k (K))$
pub struct CodifElmat {
  mass: HodgeMassElmat,
  codif: Matrix,
}
impl CodifElmat {
  pub fn new(dim: impl Into<Dim>, grade: impl Into<ExteriorGrade>) -> Self {
    let (dim, grade) = (dim.into(), grade.into());
    let mass = HodgeMassElmat::new(dim, grade);
    let codif = unit_boundary_operator(dim, grade);
    Self { mass, codif }
  }
}
impl ElMatProvider for CodifElmat {
  fn row_grade(&self) -> ExteriorGrade {
    self.mass.grade - 1
  }
  fn col_grade(&self) -> ExteriorGrade {
    self.mass.grade
  }
  fn eval(&self, metric: &Metric, chart: Chart) -> Matrix {
    let mass = self.mass.eval(metric, chart);
    &self.codif * mass
  }
}

/// Element Matrix Provider for the $(dif u, dif v)$ bilinear form.
///
/// $A = [inner(dif lambda_J, dif lambda_I)_(L^2 Lambda^(k+1) (K))]_(I,J in Delta_k (K))$
pub struct CodifDifElmat {
  mass: HodgeMassElmat,
  dif: Matrix,
  codif: Matrix,
}
impl CodifDifElmat {
  pub fn new(dim: impl Into<Dim>, grade: impl Into<ExteriorGrade>) -> Self {
    let (dim, grade) = (dim.into(), grade.into());
    let mass = HodgeMassElmat::new(dim, grade + 1);
    let dif = unit_boundary_operator(dim, grade + 1).transpose();
    let codif = dif.transpose();

    Self { mass, dif, codif }
  }
}

impl ElMatProvider for CodifDifElmat {
  fn row_grade(&self) -> ExteriorGrade {
    self.mass.grade - 1
  }
  fn col_grade(&self) -> ExteriorGrade {
    self.mass.grade - 1
  }
  fn eval(&self, metric: &Metric, chart: Chart) -> Matrix {
    let mass = self.mass.eval(metric, chart);
    &self.codif * mass * &self.dif
  }
}

/// An element integral over a cell: a quadrature rule and the mesh points its
/// nodes sit at.
///
/// The shape functions are not held here; they arrive as [`LsfSamples`] built
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
  pub fn integrate<F>(&self, shapes: &LsfSamples, chart: Chart, vol: f64, f: F) -> ElVec
  where
    F: Fn(&MeshPoint, &Tensor) -> f64,
  {
    assert_eq!(shapes.nnodes(), self.nodes.len());

    let mut elvec = ElVec::zeros(shapes.ndofs());
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
  ) -> ElMat
  where
    F: Fn(&MeshPoint, &Tensor, &Tensor) -> f64,
  {
    assert_eq!(rows.nnodes(), self.nodes.len());
    assert_eq!(cols.nnodes(), self.nodes.len());

    let mut elmat = ElMat::zeros(rows.ndofs(), cols.ndofs());
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

/// A facet of the reference cell, as $diff K$ presents it.
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

/// Quadrature over $diff K$ for an element integral: the cell's facets, each
/// integrated in the cell's chart and weighted by the sign the boundary
/// operator induces.
///
/// **Metric-free**, because the integrand is an $(n-1)$-*form* rather than a
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

  /// $integral_(diff K) omega$ of a section of grade $n-1$.
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

  /// $[integral_(diff K) f(x, W_sigma, W'_tau)]_(sigma tau)$, where `f` is the
  /// pointwise integrand of the bilinear form: at each point a bilinear map
  /// $Lambda^(k_r) times Lambda^(k_c) -> Lambda^(n-1)$, hence a section of
  /// $"Hom"(Lambda^(k_r) times.circle Lambda^(k_c), Lambda^(n-1))$ evaluated
  /// against the two shape functions.
  ///
  /// It is a family indexed by pairs of degrees of freedom, so it cannot be one
  /// [`Section`]; that is the whole difference between this and
  /// [`Self::integrate_form`].
  pub fn integrate_pair<F>(&self, rows: &LsfSamples, cols: &LsfSamples, chart: Chart, f: F) -> ElMat
  where
    F: Fn(&MeshPoint, &Tensor, &Tensor) -> Tensor,
  {
    assert_eq!(rows.nnodes(), self.nodes.len());
    assert_eq!(cols.nnodes(), self.nodes.len());

    let mut elmat = ElMat::zeros(rows.ndofs(), cols.ndofs());
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
/// The varying-coefficient counterpart of [`HodgeMassElmat`], which is exact
/// where this is a quadrature: with $alpha equiv 1$ the two agree to the
/// accuracy of the rule. Intrinsic, like every element integral here -- the
/// coefficient is a grade-0 section of the manifold, so a metric never enters
/// through it, only through the inner product on $Lambda^k$.
pub struct WeightedHodgeMassElmat<'a, F> {
  coefficient: &'a F,
  grade: ExteriorGrade,
  quad: CellQuadrature,
  shapes: LsfSamples,
}
impl<'a, F: Section> WeightedHodgeMassElmat<'a, F> {
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
impl<F: Sync + Section> ElMatProvider for WeightedHodgeMassElmat<'_, F> {
  fn row_grade(&self) -> ExteriorGrade {
    self.grade
  }
  fn col_grade(&self) -> ExteriorGrade {
    self.grade
  }
  fn eval(&self, metric: &Metric, chart: Chart) -> ElMat {
    let inner = multiform_gramian(metric, self.grade);
    self.quad.integrate_pair(
      &self.shapes,
      &self.shapes,
      chart,
      cell_volume(metric),
      |point, row, col| {
        self.coefficient.at(point).components()[0] * inner.inner(row.components(), col.components())
      },
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
/// The boundary term's star is taken in the cell's *reference* frame, and needs
/// no coherent orientation: flipping that frame flips both the star and the
/// induced orientation of $diff K$, and the product is what the term is. So
/// assembly stays independent of a gauge it must not depend on, and the
/// operator exists on a non-orientable mesh.
///
/// `velocity` is a **vector field**, not a 1-form, and nothing is sharped here:
/// $iota_v$ and $dif$ are metric-free, and the metric enters only through the
/// $L^2$ pairing and the star.
///
/// **Central and unstabilized**: each cell integrates its own trace of a shared
/// facet, so no numerical flux is chosen. Conservative at both ends of the
/// grade range, where the defect $integral_(diff K) inner(omega, eta) iota_v
/// vol$ vanishes: the shape functions are continuous at $k = 0$ and constant
/// per cell at $k = n$. Dispersive throughout -- it damps nothing, so the phase
/// error of barely resolved modes persists as oscillation, which conservation
/// does not see.
pub struct LieDerivativeElmat<'a, V> {
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

impl<'a, V: Section> LieDerivativeElmat<'a, V> {
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

impl<V: Sync + Section> ElMatProvider for LieDerivativeElmat<'_, V> {
  fn row_grade(&self) -> ExteriorGrade {
    self.grade
  }
  fn col_grade(&self) -> ExteriorGrade {
    self.grade
  }

  fn eval(&self, metric: &Metric, chart: Chart) -> ElMat {
    let inner = multiform_gramian(metric, self.grade);

    let interior = self.volume.integrate_pair(
      &self.test,
      &self.trial_dif,
      chart,
      cell_volume(metric),
      |point, test, trial_dif| {
        let advected = trial_dif.interior_product(&self.velocity.at(point));
        inner.inner(advected.components(), test.components())
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

pub type ElVec = Vector;
pub trait ElVecProvider: Sync {
  fn grade(&self) -> ExteriorGrade;
  fn eval(&self, metric: &Metric, chart: Chart) -> ElVec;
}

/// Element vector of the source load
/// $[integral_K inner(f, W_sigma)_(Lambda^k) vol]_(sigma in Delta_k (K))$.
///
/// Intrinsic: the source is a field on the manifold, the Whitney shape
/// functions are the reference ones, and both are paired in the cell's
/// reference frame under the induced inner product $Lambda^k g^(-1)$ of the
/// cell metric. Source assembly therefore runs on Regge geometry, with no
/// coordinates in sight.
pub struct SourceElVec<'a, F> {
  source: &'a F,
  quad: CellQuadrature,
  shapes: LsfSamples,
}
impl<'a, F: Section> SourceElVec<'a, F> {
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
impl<F: Sync + Section> ElVecProvider for SourceElVec<'_, F> {
  fn grade(&self) -> ExteriorGrade {
    self.source.grade()
  }
  fn eval(&self, metric: &Metric, chart: Chart) -> ElVec {
    let inner = multiform_gramian(metric, self.grade());
    self.quad.integrate(
      &self.shapes,
      chart,
      cell_volume(metric),
      |point, whitney| inner.inner(self.source.at(point).components(), whitney.components()),
    )
  }
}

#[cfg(test)]
mod test {
  use super::*;
  use multialgebra::Variance;
  use multiindex::factorial;
  use simplicial::Dim;
  use simplicial::topology::complex::Complex;

  use derham::{
    cochain::Cochain,
    interpolate::{form::WhitneyLsf, interpolant::WhitneyInterpolant},
  };
  use multialgebra::Tensor;
  use simplicial::{geometry::metric::simplex::SimplexLengthsSq, topology::simplex::unit_subsimps};

  use approx::assert_relative_eq;

  /// The single cell of the standard complex, read as a chart: what a
  /// closed-form element matrix is evaluated on when there is no mesh in sight.
  fn refchart(complex: &Complex) -> Chart<'_> {
    complex.cells().handle_iter().next().unwrap()
  }

  /// Stokes' theorem on a single cell, $integral_K dif omega = integral_(diff
  /// K) omega$, which is what the boundary quadrature has to reproduce and the
  /// only check that pins its induced signs.
  ///
  /// Metric-free on both sides: $dif$ needs none and an $(n-1)$-form over an
  /// $(n-1)$-simplex needs none either. Taking $omega$ a Whitney shape function
  /// makes the left side exact, since its differential is constant, so the
  /// identity is read against a closed form rather than a second quadrature.
  #[test]
  fn boundary_quadrature_satisfies_stokes_theorem_on_a_cell() {
    for dim in (1..=4).map(Dim::from) {
      let refcomplex = Complex::unit(dim);
      let chart = refchart(&refcomplex);
      let grade = dim - 1;
      let quadrature = BoundaryQuadrature::new(dim, Some(SimplexQuadRule::degree(dim - 1, 2)));

      let ndofs = refcomplex.nsimplices(grade);
      for (idof, dof_simp) in unit_subsimps(dim, grade).enumerate() {
        // The global Whitney form of this DOF: the interpolant of the cochain
        // that is one there and zero elsewhere.
        let mut coeffs = Vector::zeros(ndofs);
        coeffs[idof] = 1.0;
        let field = WhitneyInterpolant::new(Cochain::new(grade, coeffs), &refcomplex);

        let interior =
          WhitneyLsf::unit(dim, dof_simp).dif().components()[0] * unit_simplex_volume(dim);
        let boundary = quadrature.integrate_form(chart, &field);

        assert_relative_eq!(boundary, interior, epsilon = 1e-12);
      }
    }
  }

  /// A constant vector field in the cell's reference frame.
  struct ConstantVelocity {
    dim: Dim,
    value: Tensor,
  }
  impl Section for ConstantVelocity {
    fn dim(&self) -> Dim {
      self.dim
    }
    fn grade(&self) -> ExteriorGrade {
      Dim::ONE
    }
    fn at(&self, _point: &MeshPoint) -> Tensor {
      self.value.clone()
    }
  }

  /// $dif iota_v W_tau$ for a **constant** $v$: with $W_tau = k! sum_j (-1)^j
  /// lambda_(tau_j) beta_j$ and both $v$ and the blades $beta_j$ constant,
  /// $dif iota_v W_tau = k! sum_j (-1)^j dif lambda_(tau_j) wedge iota_v
  /// beta_j$.
  fn dif_interior_of(dim: Dim, dof_simp: Combination, v: &Tensor) -> Tensor {
    let nvertices = (dim + 1).index();
    let difbarys = unit_difbarys(dim);
    let grade = Dim::from(dof_simp.card() - 1);

    let blade =
      |c| Tensor::from_blade_signed(nvertices, multiindex::Sign::Pos, c, Variance::Covariant);
    dof_simp
      .deletions()
      .map(|(sign, vertex, rest)| {
        let difbary = blade(Combination::single(vertex)).pullback(&difbarys);
        let beta = blade(rest).pullback(&difbarys);
        sign.as_f64() * difbary.wedge(&beta.interior_product(v))
      })
      .reduce(|a, b| a + b)
      .map(|form| factorial(grade.index()) as f64 * form)
      .unwrap_or_else(|| Tensor::multiform_zero(dim, grade))
  }

  /// The identity the Lie derivative element matrix rests on: with the shape
  /// functions coclosed, integrating Cartan's second term by parts leaves it
  /// wholly on the boundary,
  /// $integral_K inner(dif iota_v omega, eta) = integral_(diff K) (iota_v
  /// omega) wedge star eta$.
  ///
  /// The left side is computed from a closed form for $dif iota_v W_tau$ at
  /// constant $v$, so the two sides share no code, and a wrong induced sign or
  /// a wrong star convention on the right cannot hide.
  #[test]
  fn cartans_second_term_is_wholly_on_the_boundary() {
    for dim in (1..=3).map(Dim::from) {
      let refcomplex = Complex::unit(dim);
      let chart = refchart(&refcomplex);
      let geo = SimplexLengthsSq::unit(dim);
      let metric = geo.metric();

      let velocity = ConstantVelocity {
        dim,
        value: Tensor::line(
          Vector::from_iterator(
            dim.index(),
            (0..dim.index()).map(|i| 0.4 * (i as f64) + 0.9),
          ),
          Variance::Contravariant,
        ),
      };

      let volume = CellQuadrature::new(dim, Some(SimplexQuadRule::degree(dim, 2)));
      let boundary = BoundaryQuadrature::new(dim, Some(SimplexQuadRule::degree(dim - 1, 2)));

      for grade in dim.range_inclusive() {
        let test = LsfSamples::whitney(dim, grade, volume.nodes());
        let inner = multiform_gramian(&metric, grade);

        let boundary_test = LsfSamples::whitney(dim, grade, boundary.nodes());
        let boundary_trial = LsfSamples::whitney(dim, grade, boundary.nodes());
        let by_parts = boundary.integrate_pair(
          &boundary_test,
          &boundary_trial,
          chart,
          |point, test, trial| {
            trial
              .interior_product(&velocity.at(point))
              .wedge(&test.star(&metric, multiindex::Sign::Pos))
          },
        );

        for (jdof, dof_simp) in unit_subsimps(dim, grade).enumerate() {
          let dif_interior = dif_interior_of(dim, dof_simp, &velocity.value);
          let direct = volume.integrate(&test, chart, cell_volume(&metric), |_point, test| {
            inner.inner(dif_interior.components(), test.components())
          });

          for idof in 0..direct.len() {
            assert_relative_eq!(by_parts[(idof, jdof)], direct[idof], epsilon = 1e-12);
          }
        }
      }
    }
  }

  /// $cal(L)_v$ annihilates a constant function: the element matrix at grade 0
  /// sends the all-ones degrees of freedom, which interpolate the constant $1$,
  /// to zero.
  ///
  /// The degenerate grade is the point. Cartan's second term vanishes there
  /// because $iota_v$ maps $Lambda^0$ into the trivial $Lambda^(-1)$, so the
  /// whole operator is $iota_v dif$ and the law reads the coboundary of a
  /// partition of unity, on the same code path every other grade takes.
  #[test]
  fn the_lie_derivative_annihilates_a_constant() {
    for dim in (1..=3).map(Dim::from) {
      let refcomplex = Complex::unit(dim);
      let chart = refchart(&refcomplex);
      let metric = SimplexLengthsSq::unit(dim).metric();

      let velocity = ConstantVelocity {
        dim,
        value: Tensor::line(
          Vector::from_iterator(
            dim.index(),
            (0..dim.index()).map(|i| 1.3 - 0.6 * (i as f64)),
          ),
          Variance::Contravariant,
        ),
      };

      let elmat = LieDerivativeElmat::new(&velocity, Dim::ZERO, 2).eval(&metric, chart);
      let constant = Vector::from_element(elmat.ncols(), 1.0);

      assert_relative_eq!((elmat * constant).norm(), 0.0, epsilon = 1e-12);
    }
  }

  /// The exact antisymmetry defect of the Lie derivative element matrix,
  /// $a_K (omega, eta) + a_K (eta, omega) = integral_(diff K) inner(omega, eta)
  /// iota_v vol$.
  ///
  /// For a constant $v$ on a flat cell $cal(L)_v$ is a derivation of the inner
  /// product and annihilates $vol$, so $inner(cal(L)_v omega, eta) +
  /// inner(omega, cal(L)_v eta) = iota_v dif inner(omega, eta)$, which Cartan
  /// and Stokes carry to the boundary. The operator is therefore skew *up to
  /// exactly this*, and on a closed manifold with a Killing field the term
  /// telescopes away and the spectrum is imaginary.
  ///
  /// This is the statement worth asserting rather than the spectrum itself: it
  /// is exact at every grade and dimension, it needs no mesh, and it says *why*
  /// the eigenvalues leave the imaginary axis instead of measuring by how much.
  #[test]
  fn the_lie_derivative_is_skew_up_to_its_boundary_term() {
    // The identity is satisfied vacuously wherever both sides vanish, as they
    // do at the top grade in one dimension, where the only Whitney 1-form is
    // constant and the two endpoints of the boundary cancel. Somewhere in the
    // sweep the defect has to be real.
    let mut largest_defect: f64 = 0.0;

    for dim in (1..=3).map(Dim::from) {
      let refcomplex = Complex::unit(dim);
      let chart = refchart(&refcomplex);
      let metric = SimplexLengthsSq::unit(dim).metric();

      let velocity = ConstantVelocity {
        dim,
        value: Tensor::line(
          Vector::from_iterator(
            dim.index(),
            (0..dim.index()).map(|i| 0.8 - 0.3 * (i as f64)),
          ),
          Variance::Contravariant,
        ),
      };
      // $iota_v vol$, the flux form the defect integrates.
      let flux = Tensor::one(dim)
        .star(&metric, multiindex::Sign::Pos)
        .interior_product(&velocity.value);

      let boundary = BoundaryQuadrature::new(dim, Some(SimplexQuadRule::degree(dim - 1, 2)));

      for grade in dim.range_inclusive() {
        let elmat = LieDerivativeElmat::new(&velocity, grade, 2).eval(&metric, chart);
        let symmetric_part = &elmat + elmat.transpose();

        let inner = multiform_gramian(&metric, grade);
        let shapes = LsfSamples::whitney(dim, grade, boundary.nodes());
        let defect = boundary.integrate_pair(&shapes, &shapes, chart, |_point, row, col| {
          inner.inner(row.components(), col.components()) * flux.clone()
        });

        largest_defect = largest_defect.max(defect.norm());
        assert_relative_eq!(&symmetric_part, &defect, epsilon = 1e-12);
      }
    }

    assert!(largest_defect > 1e-6, "the operator is not skew on a cell");
  }

  /// The varying-coefficient path against the closed form it generalizes: on a
  /// constant $alpha equiv c$ the quadrature must return $c$ times the exact
  /// [`HodgeMassElmat`], at every dimension and grade.
  ///
  /// The coefficient is a [`WhitneyInterpolant`], so nothing in this test has
  /// an embedding: the section is the interpolation of a cochain on a Regge
  /// mesh, evaluated at mesh points of the chart. Constant $c$ is what makes
  /// the closed form an oracle, and taking $c != 1$ is what catches a
  /// coefficient that is never read.
  #[test]
  fn weighted_hodge_mass_on_a_constant_is_the_closed_form() {
    for dim in (0..=3).map(Dim::from) {
      let complex = Complex::unit(dim);
      let geo = SimplexLengthsSq::unit(dim);
      let metric = geo.metric();
      let chart = refchart(&complex);

      for c in [1.0, 2.5] {
        let cochain = Cochain::constant(c, complex.skeleton_raw(Dim::ZERO));
        let coefficient = WhitneyInterpolant::new(cochain, &complex);

        for grade in dim.range_inclusive() {
          let exact = HodgeMassElmat::new(dim, grade).eval(&metric, chart);
          let quadrature =
            WeightedHodgeMassElmat::new(&coefficient, grade, Some(SimplexQuadRule::degree(dim, 2)))
              .eval(&metric, chart);
          assert_relative_eq!(&quadrature, &(c * exact), epsilon = 1e-12);
        }
      }
    }
  }

  #[test]
  fn hodge_mass0_is_scalar_mass() {
    for dim in (0..=3).map(Dim::from) {
      let geo = SimplexLengthsSq::unit(dim);
      let refcomplex = Complex::unit(dim);
      let hodge_mass =
        HodgeMassElmat::new(dim, Dim::ZERO).eval(&geo.metric(), refchart(&refcomplex));
      let metric = geo.metric();
      let scalar_mass = cell_volume(&metric) * unit_bary_gramian(dim).matrix();
      assert_relative_eq!(&hodge_mass, &scalar_mass);
    }
  }

  #[test]
  fn hodge_mass_dim2_grade1() {
    let dim = Dim::new(2);
    let grade = Dim::new(1);
    let geo = SimplexLengthsSq::unit(dim);
    let refcomplex = Complex::unit(dim);
    let computed = HodgeMassElmat::new(dim, grade).eval(&geo.metric(), refchart(&refcomplex));
    let expected = na::dmatrix![
      1./3.,1./6.,0.   ;
      1./6.,1./3.,0.   ;
      0.   ,0.   ,1./6.;
    ];
    assert_relative_eq!(&computed, &expected);
  }

  #[test]
  fn dif_n2_k1() {
    let dim = Dim::new(2);
    let grade = Dim::new(1);
    let geo = SimplexLengthsSq::unit(dim);
    let refcomplex = Complex::unit(dim);
    let computed = DifElmat::new(dim, grade).eval(&geo.metric(), refchart(&refcomplex));
    let expected = na::dmatrix![
      -1./2., 1./3.,1./6.;
      -1./2., 1./6.,1./3.;
       0.   ,-1./6.,1./6.;
    ];
    assert_relative_eq!(&computed, &expected);
  }

  #[test]
  fn codif_n2_k1() {
    let dim = Dim::new(2);
    let grade = Dim::new(1);
    let geo = SimplexLengthsSq::unit(dim);
    let refcomplex = Complex::unit(dim);
    let computed = CodifElmat::new(dim, grade).eval(&geo.metric(), refchart(&refcomplex));
    let expected = na::dmatrix![
      -1./2., -1./2., 0.   ;
       1./3.,  1./6.,-1./6.;
       1./6.,  1./3., 1./6.;
    ];
    assert_relative_eq!(&computed, &expected);
  }

  #[test]
  fn dif_dif_is_norm_of_difwhitneys() {
    for dim in (1..=3).map(Dim::from) {
      let geo = SimplexLengthsSq::unit(dim);
      let refcomplex = Complex::unit(dim);
      for grade in dim.range() {
        let difdif = CodifDifElmat::new(dim, grade).eval(&geo.metric(), refchart(&refcomplex));

        let difwhitneys: Vec<_> = unit_subsimps(dim, grade)
          .map(|simp| WhitneyLsf::unit(dim, simp).dif())
          .collect();
        let mut inner = Matrix::zeros(difwhitneys.len(), difwhitneys.len());
        for (i, awhitney) in difwhitneys.iter().enumerate() {
          for (j, bwhitney) in difwhitneys.iter().enumerate() {
            inner[(i, j)] = gramian::tensor::inner(awhitney, bwhitney, &geo.metric());
          }
        }
        inner *= geo.vol();
        assert_relative_eq!(&difdif, &inner);
      }
    }
  }
}
