//! The Galerkin assembly primitives: a load pairs as the $L^2$ functional
//! it represents (not the de Rham cochain), and every geometry source
//! (edge lengths, cell Gramians, a Lorentzian embedding) reduces to the
//! same Regge data and hence assembles identically.

use formoniq::galerkin::{BilinearForm, LinearForm};
use formoniq::operators::{SourceForm, WhitneyPairing};
use regge::lengths::CellGramians;
use regge::mesher::cartesian::CartesianGrid;
use simplicial::Dim;
use simplicial::linalg::Matrix;

/// The defining law of an assembled linear form: its pairing against a
/// discrete form is the form evaluated there, $ell(u) = sum_sigma u_sigma
/// ell(lambda_sigma)$, by linearity in the basis.
///
/// Checked against the $L^2$ pairing it represents: with the source itself a
/// Whitney form $lambda_tau$, the load is the column $tau$ of the mass, so
/// pairing it with $u$ is $u^top M e_tau$, the $L^2$ inner product of $u$
/// with $lambda_tau$. That is the statement that a Galerkin vector is the
/// $L^2$ functional and not the de Rham cochain: the two agree nowhere, and
/// a swept comparison against $u_tau$ would fail.
#[test]
fn a_load_pairs_as_the_l2_functional_it_represents() {
  use derham::{Cochain, interpolate::interpolant::WhitneyInterpolant};
  use simplicial::linalg::Vector;

  for dim in (1..=3).map(Dim::from) {
    let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
    let lengths = coords.to_edge_lengths_sq(&topology);

    for grade in dim.range_inclusive() {
      let ndofs = topology.nsimplices(grade);
      let u = Cochain::new(grade, Vector::from_fn(ndofs, |i, _| ((i % 7) as f64) - 3.0));
      // The source is the Whitney form of a fixed basis cochain, so the load
      // it induces is a known column of the mass matrix.
      let tau = ndofs / 2;
      let basis = Cochain::new(grade, Vector::from_fn(ndofs, |i, _| f64::from(i == tau)));
      let field = WhitneyInterpolant::new(basis.clone(), &topology);

      let qr = simplicial::atlas::SimplexQuadRule::degree(dim, 4);
      let load = SourceForm::new(&field, Some(qr)).assemble(&topology, &lengths);
      let mass = WhitneyPairing::mass(dim, grade).assemble(&topology, &lengths);

      let paired = load.pair(&u);
      let expected = u.coeffs().dot(&(&mass * basis.coeffs()));
      approx::assert_relative_eq!(paired, expected, epsilon = 1e-9);

      // The same field read the other way, by the de Rham map, is the basis
      // cochain itself. The load is the mass column instead, so the two
      // differ, which is what makes the law above a statement about which
      // integral was taken rather than about coefficients.
      assert!(
        (load.coeffs() - basis.coeffs()).norm() > 1e-9,
        "dim={dim} grade={grade}: the L2 load coincided with the de Rham cochain"
      );
    }
  }
}

/// Assembly consumes the edge-length primitive, so representation
/// independence is a property of the conversions into it: routing a
/// geometry through per-cell metrics
/// ([`CellGramians`]) and reading them back as edge lengths reproduces the
/// original lengths exactly, hence assembles identically. The derivation
/// chain $"lengths" -> "metric" -> "lengths"$ commutes.
#[test]
fn cell_gramians_round_trip_assembles_identically() {
  let dim = Dim::new(3);
  let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
  let lengths = coords.to_edge_lengths_sq(&topology);
  let round_trip = CellGramians::from_lengths(&topology, &lengths).to_edge_lengths_sq(&topology);

  for grade in dim.range_inclusive() {
    let from_lengths =
      Matrix::from(&WhitneyPairing::mass(dim, grade).assemble(&topology, &lengths));
    let from_round_trip =
      Matrix::from(&WhitneyPairing::mass(dim, grade).assemble(&topology, &round_trip));
    approx::assert_relative_eq!(from_lengths, from_round_trip, epsilon = 1e-12);
  }
}

/// Every geometry source reduces to the same edge-length primitive on a
/// Lorentzian mesh too: a Minkowski embedding, and the per-cell metrics it
/// induces read back as edge lengths, yield identical Regge data and hence
/// identical Galerkin matrices. This is Regge calculus doing what it was
/// invented for, a simplicial spacetime carried by edge data alone, no
/// coordinates in the assembly path.
#[test]
fn lorentzian_sources_reduce_to_the_same_regge_data() {
  use regge::coord::mesh::MeshCoords;

  for dim in (1..=3).map(Dim::from) {
    let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
    let mut matrix = coords.into_matrix();
    matrix.row_mut(0).scale_mut(0.7);
    let spacetime = MeshCoords::with_ambient(matrix, metric::Metric::minkowski(dim.index()));

    let from_coords = spacetime.to_edge_lengths_sq(&topology);
    let from_gramians = spacetime
      .to_cell_gramians(&topology)
      .to_edge_lengths_sq(&topology);

    for grade in dim.range_inclusive() {
      let a = Matrix::from(&WhitneyPairing::mass(dim, grade).assemble(&topology, &from_coords));
      let b = Matrix::from(&WhitneyPairing::mass(dim, grade).assemble(&topology, &from_gramians));
      approx::assert_relative_eq!(a, b, epsilon = 1e-12);
    }
  }
}
