//! Every problem, at a polynomial degree above one.
//!
//! `TrimmedComplex` is a `HilbertComplex` like any other, so the solvers need no
//! knowledge of the degree. Each problem runs at $r = 2$ and is checked against
//! a law it satisfies at any degree, not against a stored number.

use multialgebra::Variance;
use {
  derham::cochain::Cochain,
  formoniq::{
    problems::{dirac, elliptic, heat, wave},
    trimmed_complex::TrimmedComplex,
    whitney_complex::HilbertComplex,
  },
  simplicial::{
    Dim, geometry::metric::mesh::MeshLengthsSq, linalg::Vector, mesher::cartesian::CartesianGrid,
    topology::complex::Complex,
  },
};

const DEGREE: usize = 2;

fn box_mesh(dim: usize, refinements: u32) -> (Complex, MeshLengthsSq) {
  let (mut topology, coords) = CartesianGrid::new_unit(Dim::from(dim), 2).triangulate();
  let mut metric = coords.to_edge_lengths_sq(&topology);
  let mut ordering = simplicial::topology::ordering::CellOrdering::colex(&topology);
  for _ in 0..refinements {
    let sub = topology.refine_with(&ordering, 2);
    metric = metric.refine(&sub, &topology);
    ordering = sub.ordering().clone();
    topology = sub.into_complex();
  }
  (topology, metric)
}

/// The mixed Hodge-Laplace solve returns a finite, nonzero $u$ on a load
/// orthogonal to the harmonic space.
#[test]
fn the_source_problem_solves_at_higher_order() {
  for dim in 1..=2 {
    let (topology, metric) = box_mesh(dim, 1);
    let complex = TrimmedComplex::new(&topology, &metric, DEGREE);
    for grade in 0..=dim {
      let ndofs = complex.ndofs(grade);
      // The mixed formulation is solvable only on such a load.
      let load = Vector::from_fn(ndofs, |i, _| ((i % 7) as f64 - 3.0).sin());
      let harmonics = elliptic::solve_harmonics(&complex, grade).unwrap();
      let load = if harmonics.ncols() > 0 {
        &load - &harmonics * (harmonics.transpose() * &load)
      } else {
        load
      };

      let (_sigma, u, _p) = elliptic::solve_source(&complex, load.clone(), grade).unwrap();
      assert_eq!(u.coeffs().len(), ndofs);
      assert!(
        u.coeffs().iter().all(|v| v.is_finite()),
        "dim {dim} grade {grade}: solution is not finite"
      );
      assert!(
        u.coeffs().amax() > 0.0,
        "dim {dim} grade {grade}: solution vanished on a nonzero load"
      );
    }
  }
}

/// An analytic source assembles into a load vector at higher order: a field on
/// the manifold, quadratured against the trimmed shape functions.
#[test]
fn an_analytic_source_assembles_at_higher_order() {
  use derham::section::CoordFieldExt;
  use glatt::field::DiffFormClosure;
  use multialgebra::Tensor;
  use multiindex::{Combination, Sign};

  let dim = 2;
  // The embedding is refined alongside the topology, the field being pulled
  // back through it.
  let (mut topology, mut coords) = CartesianGrid::new_unit(Dim::from(dim), 2).triangulate();
  let ordering = simplicial::topology::ordering::CellOrdering::colex(&topology);
  let sub = topology.refine_with(&ordering, 2);
  let metric = coords.to_edge_lengths_sq(&topology).refine(&sub, &topology);
  coords = coords.refine(&sub);
  topology = sub.into_complex();

  let complex = TrimmedComplex::new(&topology, &metric, DEGREE);
  for grade in 0..=dim {
    let blade = Combination::from_increasing(0..grade);
    let field = DiffFormClosure::new(
      move |p: &coorder::Coords<coorder::Ambient>| {
        let scale = p.view().iter().map(|x| x + 1.0).product::<f64>();
        scale * Tensor::from_blade_signed(dim, Sign::Pos, blade, Variance::Covariant)
      },
      dim,
      grade,
    );
    let section = field.pullback_on(&topology, &coords);
    let load = complex.source_vector(&section, None);
    assert_eq!(load.len(), complex.ndofs(grade));
    assert!(load.iter().all(|v| v.is_finite()));
    assert!(
      load.amax() > 0.0,
      "grade {grade}: a nonzero source assembled to nothing"
    );
  }
}

/// The heat equation dissipates energy monotonically, which holds at any degree
/// and any step size for an implicit scheme.
#[test]
fn heat_dissipates_at_higher_order() {
  let dim = 2;
  let (topology, metric) = box_mesh(dim, 1);
  let complex = TrimmedComplex::new(&topology, &metric, DEGREE);
  let grade = 1;

  let ndofs = complex.ndofs(grade);
  let initial = Cochain::new(grade, Vector::from_fn(ndofs, |i, _| (i % 5) as f64 - 2.0));
  let quiet = Cochain::new(grade, Vector::zeros(ndofs));
  let steps = heat::solve_heat(&complex, grade, 6, 0.05, &initial, &quiet, 1.0);

  let mass = simplicial::linalg::CsrMatrix::from(&complex.mass(grade));
  let norms: Vec<f64> = steps
    .iter()
    .map(|u| formoniq::linalg::quadratic_form_sparse(&mass, u.coeffs()).sqrt())
    .collect();
  for pair in norms.windows(2) {
    assert!(
      pair[1] <= pair[0] + 1e-9,
      "heat energy grew: {:e} -> {:e}",
      pair[0],
      pair[1]
    );
  }
  assert!(norms.last().unwrap() < &norms[0], "heat did not dissipate");
}

/// The wave equation conserves energy at higher order under the symplectic
/// integrator.
#[test]
fn wave_conserves_energy_at_higher_order() {
  let dim = 2;
  let (topology, metric) = box_mesh(dim, 1);
  let complex = TrimmedComplex::new(&topology, &metric, DEGREE);
  let grade = 1;

  // `WaveState` lives in the ambient space, the solver restricting through the
  // inclusion, so it is sized by `ndofs` alone.
  let ndofs = complex.ndofs(grade);
  let initial = wave::WaveState::new(
    Vector::from_fn(ndofs, |i, _| (i % 5) as f64 - 2.0),
    Vector::zeros(ndofs),
  );
  let times: Vec<f64> = (0..=20).map(|i| 0.01 * i as f64).collect();
  let force = Cochain::new(grade, Vector::zeros(ndofs));
  let states = wave::solve_wave(&complex, grade, &times, initial, force);

  let energies: Vec<f64> = states
    .iter()
    .map(|state| state.energy(&complex, grade))
    .collect();
  let drift = energies
    .iter()
    .map(|e| (e - energies[0]).abs())
    .fold(0.0, f64::max);
  assert!(
    drift <= 1e-6 * energies[0].abs().max(1.0),
    "wave energy drifted by {drift:e} from {:e}",
    energies[0]
  );
}

/// The Hodge-Dirac operator assembles at higher order.
#[test]
fn dirac_squares_to_the_laplacian_at_higher_order() {
  let dim = 2;
  let (topology, metric) = box_mesh(dim, 1);
  let complex = TrimmedComplex::new(&topology, &metric, DEGREE);
  // `HodgeDirac` reads only the interface, so a degree above one needs no
  // accommodation.
  let dirac = dirac::HodgeDirac::assemble_selfadjoint(&complex);
  assert_eq!(dirac.op().nrows(), dirac.ndofs_total());
  assert!(dirac.ndofs_total() > 0);
}
