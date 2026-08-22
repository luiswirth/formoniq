//! Element-matrix laws for the operators of [`formoniq::operators`]: Stokes'
//! theorem for the boundary quadrature, the exact antisymmetry defect of the
//! Lie derivative, and the Hodge mass against its closed form.

use approx::assert_relative_eq;
use derham::{
  Cochain,
  interpolate::{form::WhitneyLsf, interpolant::WhitneyInterpolant, samples::LsfSamples},
  section::Section,
};
use formoniq::{
  galerkin::BilinearForm,
  operators::{BoundaryQuadrature, HodgeMass, LieDerivative, WeightedHodgeMass},
};
use metric::tensor::{TensorExt, inner};
use multialgebra::{Dim, ExteriorGrade, Tensor, Variance};
use nalgebra as na;
use regge::lengths::simplex::SimplexLengthsSq;
use simplicial::{
  atlas::{Chart, MeshPoint, SimplexQuadRule, unit_simplex_volume},
  linalg::Vector,
  topology::{complex::Complex, simplex::unit_subsimps},
};
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

      let interior = WhitneyLsf::unit(dim, dof_simp).dif().as_scalar() * unit_simplex_volume(dim);
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

/// The exact antisymmetry defect of the Lie derivative element matrix,
/// $a_K (omega, eta) + a_K (eta, omega) = integral_(diff K) inner(omega, eta)
/// iota_v vol$.
///
/// For a constant $v$ on a flat cell $cal(L)_v$ is a derivation of the inner
/// product and annihilates $vol$, so $inner(cal(L)_v omega, eta) +
/// inner(omega, cal(L)_v eta) = iota_v dif inner(omega, eta)$, which Cartan
/// and Stokes carry to the boundary. The operator is therefore skew up to
/// exactly this, and on a closed manifold with a Killing field the term
/// telescopes away and the spectrum is imaginary.
///
/// This is the statement worth asserting rather than the spectrum itself: it
/// is exact at every grade and dimension, it needs no mesh, and it says why
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
      let elmat = LieDerivative::new(&velocity, grade, 2).element(&metric, chart);
      let symmetric_part = &elmat + elmat.transpose();

      let shapes = LsfSamples::whitney(dim, grade, boundary.nodes());
      let defect = boundary.integrate_pair(&shapes, &shapes, chart, |_point, row, col| {
        inner(row, col, &metric) * flux.clone()
      });

      largest_defect = largest_defect.max(defect.norm());
      assert_relative_eq!(&symmetric_part, &defect, epsilon = 1e-12);
    }
  }

  assert!(largest_defect > 1e-6, "the operator is not skew on a cell");
}

/// The varying-coefficient path against the closed form it generalizes: on a
/// constant $alpha equiv c$ the quadrature must return $c$ times the exact
/// [`HodgeMass`], at every dimension and grade.
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
      let cochain = Cochain::constant(c, complex.skeleton(Dim::ZERO));
      let coefficient = WhitneyInterpolant::new(cochain, &complex);

      for grade in dim.range_inclusive() {
        let exact = HodgeMass::new(dim, grade).element(&metric);
        let quadrature =
          WeightedHodgeMass::new(&coefficient, grade, Some(SimplexQuadRule::degree(dim, 2)))
            .element(&metric, chart);
        assert_relative_eq!(&quadrature, &(c * exact), epsilon = 1e-12);
      }
    }
  }
}

/// The Whitney 1-form mass of the reference triangle against its
/// hand-computed closed form, the grade-$1$ anchor beside the grade-$0$ ones
/// of `unit_elmat`.
#[test]
fn hodge_mass_dim2_grade1() {
  let dim = Dim::new(2);
  let grade = Dim::new(1);
  let geo = SimplexLengthsSq::unit(dim);
  let computed = HodgeMass::new(dim, grade).element(&geo.metric());
  let expected = na::dmatrix![
    1./3.,1./6.,0.   ;
    1./6.,1./3.,0.   ;
    0.   ,0.   ,1./6.;
  ];
  assert_relative_eq!(&computed, &expected);
}
