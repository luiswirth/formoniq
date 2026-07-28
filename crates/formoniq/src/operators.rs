use {
  derham::{interpolate::form::WhitneyLsf, section::Section, trace::FaceTrace},
  exterior::{Covariant, Dim, ExteriorGrade, MultiForm, exterior_power, multiform_gramian},
  gramian::Metric,
  multiindex::{Combination, factorial},
  simplicial::{
    atlas::{Chart, ChartExt, MeshPoint, SimplexQuadRule, ref_difbarys, refsimp_vol},
    geometry::cell_volume,
    linalg::{Matrix, Vector},
    topology::simplex::{standard_boundary_operator, standard_subsimps},
  },
};

pub type ElMat = Matrix;
pub trait ElMatProvider: Sync {
  fn row_grade(&self) -> ExteriorGrade;
  fn col_grade(&self) -> ExteriorGrade;
  fn eval(&self, metric: &Metric, chart: Chart) -> ElMat;
}

/// Element matrix of the scalar mass bilinear form, $[integral_K lambda_i lambda_j]$.
///
/// Exact closed form: $vol(K) (1 + delta_(i j)) / ((n+1)(n+2))$.
/// The barycentric building block of the Hodge mass matrix.
fn scalar_mass_elmat(metric: &Metric) -> ElMat {
  let dim = metric.dim();
  let ndofs = dim + 1;
  let v = cell_volume(metric) / ((dim + 1) * (dim + 2)) as f64;
  let mut elmat = Matrix::from_element(ndofs, ndofs, v);
  elmat.fill_diagonal(2.0 * v);
  elmat
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
pub struct HodgeMassElmat {
  dim: Dim,
  grade: ExteriorGrade,
  simplices: Vec<Combination>,
  /// $Lambda^k$ of the reference barycentric differentials: the pullback
  /// matrix taking formal barycentric $k$-blades to reference $k$-forms.
  difbarys_power: Matrix,
}
impl HodgeMassElmat {
  pub fn new(dim: impl Into<Dim>, grade: impl Into<ExteriorGrade>) -> Self {
    let (dim, grade) = (dim.into(), grade.into());
    let simplices: Vec<_> = standard_subsimps(dim, grade).collect();
    let difbarys_power = exterior_power(&ref_difbarys(dim), grade);

    Self {
      dim,
      grade,
      simplices,
      difbarys_power,
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

    let scalar_mass = scalar_mass_elmat(metric);
    let form_gramian = multiform_gramian(metric, self.grade);

    // Inner products of the pulled-back barycentric k-blades
    // $lambda^* (e_I)$: one Cauchy-Binet sandwich for all Whitney wedge
    // terms at once.
    let blade_inners =
      &self.difbarys_power * form_gramian.matrix() * self.difbarys_power.transpose();

    let mut elmat = Matrix::zeros(self.simplices.len(), self.simplices.len());
    for (i, asimp) in self.simplices.iter().enumerate() {
      for (j, bsimp) in self.simplices.iter().enumerate() {
        let mut sum = 0.0;
        for (asign, avertex, arest) in asimp.deletions() {
          for (bsign, bvertex, brest) in bsimp.deletions() {
            sum += (asign * bsign).as_f64()
              * blade_inners[(arest.rank(), brest.rank())]
              * scalar_mass[(avertex, bvertex)];
          }
        }
        elmat[(i, j)] = sum;
      }
    }

    factorial(self.grade.index()).pow(2) as f64 * elmat
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
    let dif = standard_boundary_operator(dim, grade).transpose();
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
    let codif = standard_boundary_operator(dim, grade);
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
    let dif = standard_boundary_operator(dim, grade + 1).transpose();
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

/// The reference scaffolding of an element integral: the Whitney basis of a
/// grade, and a rule to integrate against it.
///
/// A function of $(n, k)$ alone. Every chart of the atlas is the same chart up
/// to the labelling of its vertices, so the shape functions and the quadrature
/// nodes are reference data and a cell enters only to say *where* on the
/// manifold a node lies, i.e. to make a [`MeshPoint`].
///
/// This is what an element integral with a varying coefficient needs and a
/// closed-form one does not: a coefficient is a [`Section`], evaluated
/// pointwise at those mesh points. Whether that section is analytic data pulled
/// back from a continuum or the interpolation of a cochain is invisible here,
/// which is what keeps the path intrinsic -- a discrete coefficient never
/// touches an embedding at all.
pub struct WhitneyQuadrature {
  grade: ExteriorGrade,
  qr: SimplexQuadRule,
  whitneys: Vec<WhitneyLsf>,
}
impl WhitneyQuadrature {
  /// `qr` defaults to the degree-1 Grundmann-Möller rule, the cheapest rule
  /// that is exact on affine integrands.
  pub fn new(
    dim: impl Into<Dim>,
    grade: impl Into<ExteriorGrade>,
    qr: Option<SimplexQuadRule>,
  ) -> Self {
    let (dim, grade) = (dim.into(), grade.into());
    let qr = qr.unwrap_or(SimplexQuadRule::degree(dim, 1));
    let whitneys = standard_subsimps(dim, grade)
      .map(|dof_simp| WhitneyLsf::standard(dim, dof_simp))
      .collect();
    Self {
      grade,
      qr,
      whitneys,
    }
  }

  pub fn grade(&self) -> ExteriorGrade {
    self.grade
  }
  pub fn ndofs(&self) -> usize {
    self.whitneys.len()
  }

  /// The shape functions at every quadrature node, paired with the mesh point
  /// the node sits at in `chart`.
  fn nodes(&self, chart: Chart) -> impl Iterator<Item = (MeshPoint, Vec<MultiForm>)> + use<'_> {
    let cell = chart.get().idx();
    self.qr.points().map(move |bary| {
      let values = self.whitneys.iter().map(|w| w.at_bary(bary)).collect();
      (MeshPoint::new(cell, bary.to_coords()), values)
    })
  }

  /// $[integral_K f(x, W_sigma (x)) vol]_sigma$: the element vector of a
  /// pointwise integrand read against each shape function.
  pub fn integrate<F>(&self, chart: Chart, vol: f64, f: F) -> ElVec
  where
    F: Fn(&MeshPoint, &MultiForm) -> f64,
  {
    let mut elvec = ElVec::zeros(self.ndofs());
    for ((point, values), weight) in self.nodes(chart).zip(self.qr.weights().iter()) {
      for (i, value) in values.iter().enumerate() {
        elvec[i] += weight * f(&point, value);
      }
    }
    vol * elvec
  }

  /// $[integral_K f(x, W_sigma (x), W'_tau (x)) vol]_(sigma tau)$: the element
  /// matrix of a pointwise integrand read against this basis in the rows and
  /// `cols`'s in the columns.
  ///
  /// The two bases may sit at different grades, which is what a mixed block
  /// needs. Shape functions are evaluated once per node, not once per entry.
  pub fn integrate_pair<F>(&self, cols: &Self, chart: Chart, vol: f64, f: F) -> ElMat
  where
    F: Fn(&MeshPoint, &MultiForm, &MultiForm) -> f64,
  {
    let mut elmat = ElMat::zeros(self.ndofs(), cols.ndofs());
    let col_nodes = cols.nodes(chart);
    for (((point, rows), (_, cols)), weight) in self
      .nodes(chart)
      .zip(col_nodes)
      .zip(self.qr.weights().iter())
    {
      for (i, row) in rows.iter().enumerate() {
        for (j, col) in cols.iter().enumerate() {
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
/// **Metric-free**, and the integrand's type is why. Over a cell one integrates
/// a *scalar* against $vol$, which is where a metric enters; over the boundary
/// one integrates an $(n-1)$-*form* directly, and a form over a simplex of its
/// own grade needs no metric, the geometry riding in the form's own values. So
/// the measure here is the reference one and whatever metric an integrand wants
/// -- a Hodge star, an inner product -- it reads for itself.
///
/// The integrand is an $(n-1)$-form and the quadrature takes its
/// [`FaceTrace`] onto each facet, so a caller cannot forget that only the
/// tangential part of a form is integrable over a face.
///
/// This is what a weak Lie derivative needs and a volume quadrature cannot
/// supply. Whitney shape functions are coclosed on a cell, so integrating
/// $dif iota_v omega$ by parts leaves nothing in the interior and the whole of
/// Cartan's second term on $diff K$. The facets are the cell's own, using the
/// cell's own DOFs, so the result is still an element matrix and ordinary
/// assembly scatters it; the coupling between neighbors appears because the two
/// sides of a shared facet disagree on the trace.
pub struct BoundaryQuadrature {
  dim: Dim,
  qr: SimplexQuadRule,
  facets: Vec<BoundaryFacet>,
  rows: Vec<WhitneyLsf>,
  cols: Vec<WhitneyLsf>,
}

impl BoundaryQuadrature {
  pub fn new(
    dim: impl Into<Dim>,
    row_grade: impl Into<ExteriorGrade>,
    col_grade: impl Into<ExteriorGrade>,
    qr: Option<SimplexQuadRule>,
  ) -> Self {
    let dim = dim.into();
    let facet_dim = dim - 1;
    let qr = qr.unwrap_or(SimplexQuadRule::degree(facet_dim, 1));

    let facets = Combination::full((dim + 1).index())
      .deletions()
      .map(|(sign, _, positions)| BoundaryFacet {
        sign: sign.as_f64(),
        positions,
        trace: FaceTrace::new(dim, &positions, facet_dim),
      })
      .collect();

    let shape = |grade| {
      standard_subsimps(dim, grade)
        .map(|dof_simp| WhitneyLsf::standard(dim, dof_simp))
        .collect()
    };
    Self {
      dim,
      qr,
      facets,
      rows: shape(row_grade.into()),
      cols: shape(col_grade.into()),
    }
  }

  /// $integral_(diff K) omega$ of a section of grade $n-1$.
  ///
  /// A field, not a closure: whether it is analytic data pulled back from a
  /// continuum, the interpolation of a cochain, or a combinator over either is
  /// invisible here, which is what makes natural boundary data intrinsic by the
  /// same code path that serves an embedded source.
  pub fn integrate_form(&self, chart: Chart, form: &impl Section<Covariant>) -> f64 {
    assert_eq!(form.dim(), self.dim);
    assert_eq!(
      form.grade(),
      self.dim - 1,
      "A boundary integrand is a form of grade n-1."
    );

    let mut integral = 0.0;
    for facet in &self.facets {
      for (bary, weight) in self.qr.points().zip(self.qr.weights().iter()) {
        let point = chart.point_on_face(&facet.positions, bary);
        integral += facet.sign * weight * facet.trace.top_coefficient(&form.at(&point));
      }
    }
    refsimp_vol(self.dim - 1) * integral
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
  pub fn integrate_pair<F>(&self, chart: Chart, f: F) -> ElMat
  where
    F: Fn(&MeshPoint, &MultiForm, &MultiForm) -> MultiForm,
  {
    let mut elmat = ElMat::zeros(self.rows.len(), self.cols.len());
    for facet in &self.facets {
      for (bary, weight) in self.qr.points().zip(self.qr.weights().iter()) {
        let point = chart.point_on_face(&facet.positions, bary);
        let rows: Vec<_> = self.rows.iter().map(|w| w.at_bary(point.bary())).collect();
        let cols: Vec<_> = self.cols.iter().map(|w| w.at_bary(point.bary())).collect();

        for (i, row) in rows.iter().enumerate() {
          for (j, col) in cols.iter().enumerate() {
            elmat[(i, j)] +=
              facet.sign * weight * facet.trace.top_coefficient(&f(&point, row, col));
          }
        }
      }
    }
    refsimp_vol(self.dim - 1) * elmat
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
  quad: WhitneyQuadrature,
}
impl<'a, F: Section<Covariant>> WeightedHodgeMassElmat<'a, F> {
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
    let quad = WhitneyQuadrature::new(coefficient.dim(), grade, qr);
    Self { coefficient, quad }
  }
}
impl<F: Sync + Section<Covariant>> ElMatProvider for WeightedHodgeMassElmat<'_, F> {
  fn row_grade(&self) -> ExteriorGrade {
    self.quad.grade()
  }
  fn col_grade(&self) -> ExteriorGrade {
    self.quad.grade()
  }
  fn eval(&self, metric: &Metric, chart: Chart) -> ElMat {
    let inner = multiform_gramian(metric, self.quad.grade());
    self
      .quad
      .integrate_pair(&self.quad, chart, cell_volume(metric), |point, row, col| {
        self.coefficient.at(point).coeffs()[0] * inner.inner(row.coeffs(), col.coeffs())
      })
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
  quad: WhitneyQuadrature,
}
impl<'a, F: Section<Covariant>> SourceElVec<'a, F> {
  pub fn new(source: &'a F, qr: Option<SimplexQuadRule>) -> Self {
    let quad = WhitneyQuadrature::new(source.dim(), source.grade(), qr);
    Self { source, quad }
  }
}
impl<F: Sync + Section<Covariant>> ElVecProvider for SourceElVec<'_, F> {
  fn grade(&self) -> ExteriorGrade {
    self.source.grade()
  }
  fn eval(&self, metric: &Metric, chart: Chart) -> ElVec {
    let inner = multiform_gramian(metric, self.grade());
    self
      .quad
      .integrate(chart, cell_volume(metric), |point, whitney| {
        inner.inner(self.source.at(point).coeffs(), whitney.coeffs())
      })
  }
}

#[cfg(test)]
mod test {
  use super::*;
  use simplicial::Dim;
  use simplicial::topology::complex::Complex;

  use derham::{
    cochain::Cochain,
    interpolate::{form::WhitneyLsf, interpolant::WhitneyInterpolant},
  };
  use simplicial::{
    geometry::metric::simplex::SimplexLengthsSq, topology::simplex::standard_subsimps,
  };

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
      let refcomplex = Complex::standard(dim);
      let chart = refchart(&refcomplex);
      let grade = dim - 1;
      let quadrature =
        BoundaryQuadrature::new(dim, grade, grade, Some(SimplexQuadRule::degree(dim - 1, 2)));

      let ndofs = refcomplex.nsimplices(grade);
      for (idof, dof_simp) in standard_subsimps(dim, grade).enumerate() {
        // The global Whitney form of this DOF: the interpolant of the cochain
        // that is one there and zero elsewhere.
        let mut coeffs = Vector::zeros(ndofs);
        coeffs[idof] = 1.0;
        let field = WhitneyInterpolant::new(Cochain::new(grade, coeffs), &refcomplex);

        let interior = WhitneyLsf::standard(dim, dof_simp).dif().coeffs()[0] * refsimp_vol(dim);
        let boundary = quadrature.integrate_form(chart, &field);

        assert_relative_eq!(boundary, interior, epsilon = 1e-12);
      }
    }
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
      let complex = Complex::standard(dim);
      let geo = SimplexLengthsSq::standard(dim);
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
      let geo = SimplexLengthsSq::standard(dim);
      let refcomplex = Complex::standard(dim);
      let hodge_mass =
        HodgeMassElmat::new(dim, Dim::ZERO).eval(&geo.metric(), refchart(&refcomplex));
      let scalar_mass = scalar_mass_elmat(&geo.metric());
      assert_relative_eq!(&hodge_mass, &scalar_mass);
    }
  }

  #[test]
  fn hodge_mass_dim2_grade1() {
    let dim = Dim::new(2);
    let grade = Dim::new(1);
    let geo = SimplexLengthsSq::standard(dim);
    let refcomplex = Complex::standard(dim);
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
    let geo = SimplexLengthsSq::standard(dim);
    let refcomplex = Complex::standard(dim);
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
    let geo = SimplexLengthsSq::standard(dim);
    let refcomplex = Complex::standard(dim);
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
      let geo = SimplexLengthsSq::standard(dim);
      let refcomplex = Complex::standard(dim);
      for grade in dim.range() {
        let difdif = CodifDifElmat::new(dim, grade).eval(&geo.metric(), refchart(&refcomplex));

        let difwhitneys: Vec<_> = standard_subsimps(dim, grade)
          .map(|simp| WhitneyLsf::standard(dim, simp).dif())
          .collect();
        let mut inner = Matrix::zeros(difwhitneys.len(), difwhitneys.len());
        for (i, awhitney) in difwhitneys.iter().enumerate() {
          for (j, bwhitney) in difwhitneys.iter().enumerate() {
            inner[(i, j)] = awhitney.inner(bwhitney, &geo.metric());
          }
        }
        inner *= geo.vol();
        assert_relative_eq!(&difdif, &inner);
      }
    }
  }
}
