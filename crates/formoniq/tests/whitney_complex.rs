//! Laws for [`formoniq::whitney_complex`]: the codifferential is the
//! $L^2$-adjoint of the exterior derivative and is nilpotent, the energy
//! and full Hodge-Dirac norms decompose the way they are defined to, the
//! de Rham operators are total at the trivial ends of the grade range, and
//! the $L^2$ pairing is genuinely distinct from the chain-cochain pairing.

use derham::Cochain;
use formoniq::{
  linalg::quadratic_form_sparse,
  whitney_complex::{HilbertComplex, WhitneyComplex, l2_pairing},
};
use multialgebra::ExteriorGrade;
use regge::mesher::cartesian::CartesianGrid;
use simplicial::{Dim, linalg::Vector, topology::complex::Complex};

/// The stiffness is one matrix with two routes to it: the element-local
/// sandwich that [`HilbertComplex::dif_both`] assembles, and the global
/// product $D^T M_(k+1) D$ of three separately assembled matrices.
///
/// They agree because the exterior derivative of a Whitney form is the
/// coboundary of the reference cell, so the contraction commutes with the
/// scatter. Swept over every dimension and grade, the extremes included:
/// at the top grade both sides are the zero operator, which is the case a
/// route that special-cased the empty codomain would get wrong.
#[test]
fn dif_both_local_and_global_routes_agree() {
  for dim in (1..=3).map(Dim::from) {
    let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
    let lengths = coords.to_edge_lengths_sq(&topology);
    let whitney = WhitneyComplex::new(&topology, &lengths);

    for grade in dim.range_inclusive() {
      let local = whitney.dif_both(grade + 1);

      let dif = whitney.dif(grade);
      let mass = whitney.mass(grade + 1);
      let global = dif.transpose() * mass * dif;

      assert_eq!(local.nrows(), global.nrows());
      assert_eq!(local.ncols(), global.ncols());
      let residual = (&local - &global)
        .values()
        .iter()
        .fold(0.0f64, |acc, v| acc.max(v.abs()));
      let scale = local
        .values()
        .iter()
        .fold(0.0f64, |acc, v| acc.max(v.abs()));
      assert!(
        residual <= 1e-12 * scale.max(1.0),
        "dim {dim:?} grade {grade:?}: routes differ by {residual:e}"
      );
      // The law must be able to fail: at every grade below the top the
      // stiffness is a nonzero operator, so agreement is not two zeros
      // matching.
      if grade < dim {
        assert!(
          scale > 1e-6,
          "dim {dim:?} grade {grade:?}: stiffness vanished"
        );
      }
    }
  }
}

/// The weak codifferential has the same two routes as the stiffness, and
/// they agree: the element-local pairing that [`HilbertComplex::dif_test`]
/// assembles, and the global product $(D^(k-1))^T M_k$ of two assembled
/// matrices. Swept over every dimension and grade, the degenerate grade $0$
/// included, where the $sigma$ space is empty and both are the $0 times
/// "ndofs"(0)$ matrix.
#[test]
fn codif_local_and_global_routes_agree() {
  for dim in (1..=3).map(Dim::from) {
    let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
    let lengths = coords.to_edge_lengths_sq(&topology);
    let whitney = WhitneyComplex::new(&topology, &lengths);

    for grade in dim.range_inclusive() {
      let local = whitney.dif_test(grade);
      let global = whitney.dif(grade - 1).transpose() * &whitney.mass(grade);

      assert_eq!(local.nrows(), global.nrows());
      assert_eq!(local.ncols(), global.ncols());
      let residual = (&local - &global)
        .values()
        .iter()
        .fold(0.0f64, |acc, v| acc.max(v.abs()));
      let scale = local
        .values()
        .iter()
        .fold(0.0f64, |acc, v| acc.max(v.abs()));
      assert!(
        residual <= 1e-12 * scale.max(1.0),
        "dim {dim:?} grade {grade:?}: routes differ by {residual:e}"
      );
      // Above grade 0 the sigma space is nonempty and the coupling is a
      // nonzero matrix, so the law is not two empties agreeing.
      if grade > 0 {
        assert!(
          scale > 1e-6,
          "dim {dim:?} grade {grade:?}: coupling vanished"
        );
      }
    }
  }
}

/// The relative stiffness is the restriction of the full one, which is the
/// subcomplex property in matrix form and is why the relative complex never
/// has to assemble a mass one grade up.
///
/// Checked against the definition it replaces, $D_"rel"^T M_"rel" D_"rel"$
/// built from the relative operators, on a mesh with a genuine boundary so
/// that the inclusion is not the identity and the law can fail.
#[test]
fn the_relative_stiffness_is_the_restriction_of_the_full_one() {
  for dim in (1..=3).map(Dim::from) {
    let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
    let lengths = coords.to_edge_lengths_sq(&topology);
    let relative = WhitneyComplex::new(&topology, &lengths).relative();

    for grade in dim.range_inclusive() {
      // Below the top the boundary carries simplices of that grade and the
      // inclusion is a proper one, so the law is not two identities agreeing.
      // At the top there are none: the boundary subcomplex has dimension
      // $n - 1$, so no cell is constrained and the relative complex coincides
      // with the full one there.
      if grade < dim {
        assert!(relative.ndofs(grade) < topology.nsimplices(grade));
      }

      let restricted = relative.dif_both(grade + 1);

      let dif = relative.dif(grade);
      let mass = relative.mass(grade + 1);
      let by_definition = dif.transpose() * mass * dif;

      let residual = (&restricted - &by_definition)
        .values()
        .iter()
        .fold(0.0f64, |acc, v| acc.max(v.abs()));
      let scale = restricted
        .values()
        .iter()
        .fold(0.0f64, |acc, v| acc.max(v.abs()));
      assert!(
        residual <= 1e-12 * scale.max(1.0),
        "dim {dim:?} grade {grade:?}: differs by {residual:e}"
      );
    }
  }
}

/// The full $H Lambda(dif)$ norm is the Pythagorean sum of the $L^2$ norm and
/// the $dif$ seminorm, and its Gram matrix [`HilbertComplex::hdif_gram`]
/// realizes it as a quadratic form: two views of one inner product.
#[test]
fn hdif_norm_and_gram_agree() {
  for dim in (1..=3).map(Dim::from) {
    let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
    let lengths = coords.to_edge_lengths_sq(&topology);
    let whitney = WhitneyComplex::new(&topology, &lengths);

    for grade in dim.range_inclusive() {
      let ndofs = topology.nsimplices(grade);
      let u = Cochain::new(
        grade,
        Vector::from_iterator(ndofs, (0..ndofs).map(|i| ((i % 5) as f64) - 2.0)),
      );

      let full = whitney.norm_hdif(&u);
      let pythag = (whitney.norm_l2(&u).powi(2) + whitney.seminorm_hdif(&u).powi(2)).sqrt();
      let gram = quadratic_form_sparse(&whitney.hdif_gram(grade), u.coeffs()).sqrt();

      assert!((full - pythag).abs() < 1e-12, "dim={dim} grade={grade}");
      assert!(
        (full - gram).abs() < 1e-10,
        "dim={dim} grade={grade}: {full} vs {gram}"
      );
      assert!(
        full >= whitney.seminorm_hdif(&u) - 1e-12,
        "full norm dominates seminorm"
      );
    }
  }
}

fn sample(grade: ExteriorGrade, topology: &Complex) -> Cochain {
  let ndofs = topology.nsimplices(grade);
  Cochain::new(
    grade,
    Vector::from_iterator(ndofs, (0..ndofs).map(|i| ((i * 3 % 7) as f64) - 3.0)),
  )
}

/// The defining law of the codifferential: it is the $L^2$-adjoint of $dif$,
/// $angle.l delta u, tau angle.r_(k-1) = angle.l u, dif tau angle.r_k$ for
/// every $tau in Lambda^(k-1)$. Swept over dimension and grade.
#[test]
fn codif_is_the_adjoint_of_dif() {
  use formoniq::linalg::bilinear_form_sparse;
  for dim in (1..=3).map(Dim::from) {
    let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
    let lengths = coords.to_edge_lengths_sq(&topology);
    let whitney = WhitneyComplex::new(&topology, &lengths);

    for grade in Dim::ONE.range_to_inclusive(dim) {
      let u = sample(grade, &topology);
      let tau = sample(grade - 1, &topology);
      let sigma = whitney.codif_cochain(&u);

      let mass_lower = whitney.mass(grade - 1);
      let mass_k = whitney.mass(grade);
      let lhs = bilinear_form_sparse(&mass_lower, sigma.coeffs(), tau.coeffs());
      let rhs = bilinear_form_sparse(&mass_k, u.coeffs(), tau.dif(&topology).coeffs());

      assert!(
        (lhs - rhs).abs() < 1e-9,
        "dim={dim} grade={grade}: {lhs} vs {rhs}"
      );
    }
  }
}

/// $delta compose delta = 0$: the codifferential is nilpotent, dual to
/// $dif compose dif = 0$. Needs grade $>= 2$ so both codifferentials land in a
/// real space.
#[test]
fn codif_is_nilpotent() {
  for dim in (2..=3).map(Dim::from) {
    let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
    let lengths = coords.to_edge_lengths_sq(&topology);
    let whitney = WhitneyComplex::new(&topology, &lengths);

    for grade in Dim::new(2).range_to_inclusive(dim) {
      let u = sample(grade, &topology);
      let ddu = whitney.codif_cochain(&whitney.codif_cochain(&u));
      assert!(whitney.norm_l2(&ddu) < 1e-9, "dim={dim} grade={grade}");
    }
  }
}

/// The energy and full Hodge-Dirac norms decompose as the Pythagorean sums
/// they are defined to be, total over every grade including the degenerate
/// $0$ and $n$ where a seminorm vanishes.
#[test]
fn delta_norms_are_total_and_pythagorean() {
  for dim in (1..=3).map(Dim::from) {
    let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
    let lengths = coords.to_edge_lengths_sq(&topology);
    let whitney = WhitneyComplex::new(&topology, &lengths);

    for grade in dim.range_inclusive() {
      let u = sample(grade, &topology);
      let (l2, hd, hcd) = (
        whitney.norm_l2(&u),
        whitney.seminorm_hdif(&u),
        whitney.seminorm_hcodif(&u),
      );
      assert!((whitney.seminorm_energy(&u) - (hd * hd + hcd * hcd).sqrt()).abs() < 1e-12);
      assert!((whitney.norm_full(&u) - (l2 * l2 + hd * hd + hcd * hcd).sqrt()).abs() < 1e-12);
      if grade == 0 {
        assert_eq!(whitney.seminorm_hcodif(&u), 0.0, "delta = 0 at grade 0");
        // delta u is the empty cochain of the trivial space Lambda^(-1) = 0,
        // not a missing value.
        let du = whitney.codif_cochain(&u);
        assert_eq!(du.grade(), Dim::new(-1));
        assert_eq!(du.coeffs().len(), 0);
      }
    }
  }
}

/// The de Rham operators are total at the trivial ends: a degree past either
/// end of the complex names the zero space $Lambda^k = 0$ ($k in.not [0, n]$),
/// so every accessor returns the correctly-shaped empty object rather than
/// panicking. This is the $Z$-graded degree cashed out, one step past each
/// end runs the same code and returns the mathematically trivial answer.
#[test]
fn operators_are_total_at_the_trivial_ends() {
  for dim in (1..=3).map(Dim::from) {
    let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
    let lengths = coords.to_edge_lengths_sq(&topology);
    let whitney = WhitneyComplex::new(&topology, &lengths);

    let ndofs0 = whitney.ndofs(Dim::ZERO);
    let ndofs_top = whitney.ndofs(dim);

    for ghost in [Dim::new(-1), dim + 1] {
      assert_eq!(whitney.ndofs(ghost), 0, "dim={dim} ghost={ghost}");
      assert_eq!(whitney.harmonic_dim(ghost), 0);

      let mass = whitney.mass(ghost);
      assert_eq!((mass.nrows(), mass.ncols()), (0, 0));

      let ldd = whitney.dif_both(ghost + 1);
      assert_eq!((ldd.nrows(), ldd.ncols()), (0, 0));

      let incl = HilbertComplex::inclusion(&whitney, ghost);
      assert_eq!((incl.nrows(), incl.ncols()), (0, 0));
    }

    // $dif$ at each end keeps one honest zero dimension: $dif^(-1): 0 ->
    // Lambda^0$ is $"ndofs"(0) times 0$, the top $dif^n: Lambda^n -> 0$ is
    // $0 times "ndofs"(n)$.
    let d_below = whitney.dif(Dim::new(-1));
    assert_eq!((d_below.nrows(), d_below.ncols()), (ndofs0, 0));
    let d_top = whitney.dif(dim);
    assert_eq!((d_top.nrows(), d_top.ncols()), (0, ndofs_top));

    // The top-grade stiffness is the honest $"ndofs"(n)^2$ zero operator, from
    // the general formula with no special case.
    let ldd_top = whitney.dif_both(dim + 1);
    assert_eq!((ldd_top.nrows(), ldd_top.ncols()), (ndofs_top, ndofs_top));

    assert!(ldd_top.values().iter().all(|&v| v == 0.0), "dim={dim}");
  }
}

/// The $L^2$ pairing is symmetric and positive definite on a Riemannian
/// geometry, and it is not the chain-cochain pairing.
///
/// The two dualities a discrete complex carries, kept apart. The metric-free
/// one integrates a cochain over a chain and needs only the incidence; this
/// one needs the mass matrix, hence a geometry. Asserting they disagree on a
/// nontrivial input is what stops a later refactor from quietly routing one
/// through the other.
#[test]
fn the_l2_pairing_is_the_metric_duality_and_the_other_is_not() {
  use approx::assert_relative_eq;
  use derham::pairing;
  use simplicial::topology::chain::Chain;

  for dim in 1..=3 {
    let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
    let geometry = coords.to_edge_lengths_sq(&topology);
    let complex = WhitneyComplex::new(&topology, &geometry);

    for grade in 0..=dim {
      let ndofs = complex.ndofs(grade);
      let u = Cochain::new(grade, Vector::from_fn(ndofs, |i, _| ((i % 5) as f64) - 2.0));
      let v = Cochain::new(grade, Vector::from_fn(ndofs, |i, _| ((i % 7) as f64) - 3.0));

      assert_relative_eq!(
        l2_pairing(&complex, &u, &v),
        l2_pairing(&complex, &v, &u),
        epsilon = 1e-10
      );
      assert!(
        l2_pairing(&complex, &u, &u) > 0.0,
        "dim {dim} grade {grade}: the L2 pairing is not positive definite"
      );

      // The same cochain against the chain with those coefficients: no
      // geometry consulted, and a different number.
      let chain = Chain::from_vec(grade, v.coeffs().iter().map(|c| c.round() as i64).collect());
      let combinatorial = pairing(&u, &chain.extend_scalars(|&c| c as f64));
      assert!(
        (combinatorial - l2_pairing(&complex, &u, &v)).abs() > 1e-9,
        "dim {dim} grade {grade}: the two pairings coincide, so one is not what it claims"
      );
    }
  }
}
