//! Laws for [`formoniq::problems::dirac`]: the mixed Hodge-Dirac operator
//! is self-adjoint and its square is the Hodge-Laplacian block by block,
//! the leapfrog integrator conserves the staggered energy of the
//! Hodge-Dirac system, and the source and harmonic-projection solves
//! reproduce their defining identities.

use approx::assert_relative_eq;
use derham::Cochain;
use formoniq::{
  hodge::HodgeBlocks,
  linalg::faer::FaerCholesky,
  problems::dirac::{
    HodgeDirac, MixedField, solve_dirac, solve_dirac_leapfrog, solve_dirac_source, top_harmonic,
  },
  time::Leapfrog,
  whitney_complex::WhitneyComplex,
};
use regge::mesher::cartesian::CartesianGrid;
use simplicial::{Dim, linalg::Vector};

/// A deterministic full field: every grade populated with a reproducible
/// pattern, enough to couple all rungs of the complex.
fn seed_field(dirac: &HodgeDirac) -> MixedField {
  let grades = dirac
    .dim()
    .range_inclusive()
    .map(|k| {
      let n = dirac.block(k).1;
      Cochain::new(
        k,
        Vector::from_fn(n, |i, _| ((7 * i + 3 * k.index() + 1) % 11) as f64 - 5.0),
      )
    })
    .collect();
  MixedField::new(grades)
}

/// Poincaré--Lefschetz, discretely: the closed-form top-grade harmonic
/// $h_n = M_n^(-1) z$ is annihilated by the massless self-adjoint
/// Hodge--Dirac operator, in every dimension and on either signature. This is
/// the statement that $ker(dif + delta) supset.eq H^n (M, diff M)$ is realized
/// exactly on the nose by the fundamental class, with no eigensolve.
#[test]
fn top_harmonic_is_annihilated() {
  for dim in (1..=4).map(Dim::from) {
    for minkowski in [false, true] {
      let (topology, coords) = if minkowski {
        CartesianGrid::minkowski(dim, 2)
      } else {
        CartesianGrid::new_unit(dim, 2).triangulate()
      };
      let regge = coords.to_edge_lengths_sq(&topology);
      let whitney = WhitneyComplex::new(&topology, &regge);
      let relative = whitney.relative();

      let h = top_harmonic(&relative).expect("a box is orientable");
      let dirac = HodgeDirac::assemble_selfadjoint(&relative);
      let residual = (dirac.op() * &h).norm() / h.norm();

      assert!(
        residual < 1e-9,
        "dim {dim} minkowski {minkowski}: |A h|/|h| = {residual:.3e}"
      );
    }
  }
}

/// The discrete codifferential is the adjoint of the exterior derivative:
/// the assembled Hodge–Dirac operator is skew-symmetric, $A + A^T = 0$, to
/// roundoff and at every dimension. This is integration by parts made
/// structural, the super-diagonal blocks are the negated transposes of the
/// sub-diagonal ones by construction, and it is what conserves energy.
#[test]
fn operator_is_skew_symmetric() {
  for dim in (1..=3).map(Dim::from) {
    let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
    let metric = coords.to_edge_lengths_sq(&topology);
    let whitney = WhitneyComplex::new(&topology, &metric);
    let dirac = HodgeDirac::assemble(&whitney);

    let a = dirac.op();
    let skew = a + &a.transpose();
    assert_relative_eq!(
      skew.values().iter().fold(0.0, |m: f64, &v| m.max(v.abs())),
      0.0
    );
  }
}

/// The defining Dirac law: $sans(D)^2 = -Delta$. The discrete Hodge–Dirac
/// operator $M^(-1) A = dif - delta_h$ squared equals the negative discrete
/// Hodge–Laplacian, grade by grade, the grade-shifting-by-two terms
/// canceling by $dif compose dif = 0$. Checked against the independently
/// assembled up/down Laplacian blocks of [`HodgeBlocks`], at every grade.
#[test]
fn dirac_squared_is_negative_hodge_laplacian() {
  for dim in (1..=3).map(Dim::from) {
    let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
    let metric = coords.to_edge_lengths_sq(&topology);
    let whitney = WhitneyComplex::new(&topology, &metric);
    let dirac = HodgeDirac::assemble(&whitney);

    // Mass solves for $M^(-1)$, one factorization per grade.
    let chol: Vec<_> = dim
      .range_inclusive()
      .map(|k| FaerCholesky::new(dirac.mass(k).clone()))
      .collect();
    // The strong Hodge–Dirac action $v = (dif - delta_h) u$, from $M v = A u$.
    let apply_dirac = |u: &MixedField| {
      let mv = dirac.op() * dirac.flatten(u);
      let grades = dim
        .range_inclusive()
        .map(|k| {
          let (off, n) = dirac.block(k);
          Cochain::new(k, chol[k.index()].solve(&mv.rows(off, n).into_owned()))
        })
        .collect();
      MixedField::new(grades)
    };

    let u = seed_field(&dirac);
    let d2u = apply_dirac(&apply_dirac(&u));

    #[allow(clippy::needless_range_loop)] // grade is the mathematical index
    for grade in dim.range_inclusive() {
      // The Hodge–Laplacian $Delta_h u|_k = M_k^(-1)(K^"up" + K^"dn") u_k$.
      let hb = HodgeBlocks::compute(&whitney, grade);
      let uk = u.grade(grade).coeffs();
      let up = &hb.dif_both * uk;
      let dn = &hb.dif_test.transpose() * hb.codif(uk);
      let lap = chol[grade.index()].solve(&(up + dn));

      let lhs = d2u.grade(grade).coeffs();
      assert_relative_eq!(
        (lhs + &lap).norm(),
        0.0,
        epsilon = 1e-9 * lap.norm().max(1.0)
      );
    }
  }
}

/// The structure-preserving law, at every dimension: the total Hodge–Dirac
/// energy $1/2 norm(u)_(L^2)^2 = 1/2 thin u^T M u$ is conserved to roundoff.
/// Gauss–Legendre is symplectic and, on this linear skew system, conserves the
/// quadratic invariant exactly, across all grades of the coupled complex
/// at once.
#[test]
fn energy_conserved_at_every_dimension() {
  for dim in (1..=3).map(Dim::from) {
    let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
    let metric = coords.to_edge_lengths_sq(&topology);
    let whitney = WhitneyComplex::new(&topology, &metric);
    let dirac = HodgeDirac::assemble(&whitney);

    let initial = seed_field(&dirac);
    let times: Vec<f64> = (0..=100).map(|i| 0.05 * i as f64).collect();
    let solution = solve_dirac(&whitney, &times, initial);

    let energy0 = dirac.energy(&solution[0]);
    assert!(energy0 > 0.0);
    for state in &solution {
      let energy = dirac.energy(state);
      assert_relative_eq!(energy, energy0, epsilon = 1e-9 * energy0);
    }
  }
}

/// The explicit leapfrog is structure-preserving too. Built from the same
/// Hodge–Dirac $M$ and skew $A$, 2-colored by grade parity, it conserves its
/// staggered invariant to roundoff at every dimension, within CFL. The same
/// symplectic guarantee as Gauss–Legendre, for the cheap explicit scheme. This
/// also exercises the coloring: [`Leapfrog::new`] asserts $A$ is
/// block-antidiagonal under it, which holds iff $dif, delta$ never couple two
/// grades of the same parity.
#[test]
fn leapfrog_conserves_staggered_energy_at_every_dimension() {
  for dim in (1..=3).map(Dim::from) {
    let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
    let metric = coords.to_edge_lengths_sq(&topology);
    let whitney = WhitneyComplex::new(&topology, &metric);
    let dirac = HodgeDirac::assemble(&whitney);
    let color = dirac.grade_parity_coloring();

    // Within the CFL limit (wave speed c = 1 in vacuum).
    let dt = 0.1 * metric.mesh_width_min();
    let leapfrog = Leapfrog::new(dirac.mass_block(), dirac.op(), &color, dt);

    let mut y = dirac.flatten(&seed_field(&dirac));
    let e0 = leapfrog.conserved_energy(&y);
    assert!(e0 > 0.0);
    for _ in 0..200 {
      y = leapfrog.step(&y);
      assert_relative_eq!(leapfrog.conserved_energy(&y), e0, epsilon = 1e-9 * e0);
    }
  }
}

/// The canonical Hodge–Dirac operator is self-adjoint: $A = A^T$ to
/// roundoff, on Riemannian and Lorentzian geometry alike, the covariant
/// counterpart of [`operator_is_skew_symmetric`], the one sign flipped.
#[test]
fn selfadjoint_operator_is_symmetric() {
  for dim in (1..=3).map(Dim::from) {
    let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
    let (_, spacetime) = CartesianGrid::minkowski(dim, 2);
    let riemannian = coords.to_edge_lengths_sq(&topology);
    let lorentzian = spacetime.to_edge_lengths_sq(&topology);

    let check = |dirac: &HodgeDirac| {
      let a = dirac.op();
      let sym = a - &a.transpose();
      assert_relative_eq!(
        sym.values().iter().fold(0.0, |m: f64, &v| m.max(v.abs())),
        0.0
      );
    };
    check(&HodgeDirac::assemble_selfadjoint(&WhitneyComplex::new(
      &topology,
      &riemannian,
    )));
    check(&HodgeDirac::assemble_selfadjoint(&WhitneyComplex::new(
      &topology,
      &lorentzian,
    )));
  }
}

/// The defining Dirac law in its covariant form, on a Lorentzian spacetime
/// mesh: $sans(D)^2 = Delta$, the discrete $dif + delta$ squared equals
/// the discrete Hodge–Laplacian, which on Minkowski geometry is the
/// d'Alembertian, hyperbolic through the signature alone. Same
/// grade-by-grade check as [`dirac_squared_is_negative_hodge_laplacian`],
/// with the sign flipped and every solve LU, since the Lorentzian masses
/// are symmetric indefinite, not s.p.d.
#[test]
fn selfadjoint_dirac_squares_to_hodge_laplacian_on_minkowski() {
  for dim in (1..=3).map(Dim::from) {
    let (topology, spacetime) = CartesianGrid::minkowski(dim, 2);
    let regge = spacetime.to_edge_lengths_sq(&topology);
    let whitney = WhitneyComplex::new(&topology, &regge);
    let dirac = HodgeDirac::assemble_selfadjoint(&whitney);

    let lu: Vec<_> = dim
      .range_inclusive()
      .map(|k| formoniq::linalg::faer::FaerLu::new(dirac.mass(k).clone()))
      .collect();
    let apply_dirac = |u: &MixedField| {
      let mv = dirac.op() * dirac.flatten(u);
      let grades = dim
        .range_inclusive()
        .map(|k| {
          let (off, n) = dirac.block(k);
          Cochain::new(k, lu[k.index()].solve(&mv.rows(off, n).into_owned()))
        })
        .collect();
      MixedField::new(grades)
    };

    let u = seed_field(&dirac);
    let d2u = apply_dirac(&apply_dirac(&u));

    #[allow(clippy::needless_range_loop)] // grade is the mathematical index
    for grade in dim.range_inclusive() {
      let hb = HodgeBlocks::compute(&whitney, grade);
      let uk = u.grade(grade).coeffs();
      let up = &hb.dif_both * uk;
      let dn = if hb.n_sigma > 0 {
        let s =
          formoniq::linalg::faer::FaerLu::new(hb.mass_sigma.clone()).solve(&(&hb.dif_test * uk));
        &hb.dif_test.transpose() * s
      } else {
        Vector::zeros(hb.n_u)
      };
      let lap = lu[grade.index()].solve(&(up + dn));

      let lhs = d2u.grade(grade).coeffs();
      assert_relative_eq!(
        (lhs - &lap).norm(),
        0.0,
        epsilon = 1e-9 * lap.norm().max(1.0)
      );
    }
  }
}

/// [`solve_dirac_source`] reproduces a solution that lies in the Whitney
/// space exactly: a constant mixed-grade form $u$ has $dif u = delta u = 0$,
/// so $(sans(D) + m) u = m u$, and with load $m M u$ and essential data $u$
/// the discrete solution is $u$ itself to solver precision, on Riemannian
/// and Lorentzian geometry alike.
#[test]
fn dirac_source_reproduces_constant_field() {
  use regge::coord::mesh::MeshCoords;
  use regge::lengths::mesh::MeshLengthsSq;
  use simplicial::topology::complex::Complex;

  fn run(topology: &Complex, coords: &MeshCoords, geometry: &MeshLengthsSq) {
    let dim = topology.dim();
    let whitney = WhitneyComplex::new(topology, geometry);
    let relative = whitney.relative();
    let dirac = HodgeDirac::assemble_selfadjoint(&whitney);

    // The de Rham coefficients of the constant form with every component
    // 1 on every grade: integrals of $sum_I dif x^I$ over the simplices.
    let exact = MixedField::new(
      dim
        .range_inclusive()
        .map(|k| {
          let form = glatt::field::DiffFormClosure::new(
            move |_: &coorder::Coord| {
              multialgebra::Tensor::multiform(
                Vector::from_element(multialgebra::exterior_dim(dim, k), 1.0),
                dim,
                k,
              )
            },
            dim,
            k,
          );
          let field = derham::section::CoordFieldExt::pullback_on(&form, topology, coords);
          derham::project::derham_map(&field, topology, 2)
        })
        .collect(),
    );

    let mass_term = 1.0;
    let load = dirac.unflatten(&(dirac.mass_block() * dirac.flatten(&exact) * mass_term));
    let solution = solve_dirac_source(&relative, mass_term, &load, &exact);

    for k in dim.range_inclusive() {
      assert_relative_eq!(
        solution.grade(k).coeffs(),
        exact.grade(k).coeffs(),
        epsilon = 1e-9
      );
    }
  }

  for dim in (1..=3).map(Dim::from) {
    let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
    let riemannian = coords.to_edge_lengths_sq(&topology);
    run(&topology, &coords, &riemannian);

    // On the Minkowski side the constant field is interpolated on the
    // causally generic (time-scaled) coordinates themselves.
    let (topology, spacetime) = CartesianGrid::minkowski(dim, 2);
    let euclidean_view = MeshCoords::new(spacetime.matrix().clone());
    run(
      &topology,
      &euclidean_view,
      &spacetime.to_edge_lengths_sq(&topology),
    );
  }
}

/// The explicit and implicit solvers integrate the same equation: over a
/// short run at small $dif t$ they agree to the leapfrog's second order. This
/// validates the [`solve_dirac_leapfrog`] wiring (restrict/extend, flatten,
/// grade-parity coloring) against the trusted Gauss–Legendre [`solve_dirac`].
#[test]
fn leapfrog_agrees_with_gauss_legendre() {
  let (topology, coords) = CartesianGrid::new_unit(Dim::new(2), 2).triangulate();
  let metric = coords.to_edge_lengths_sq(&topology);
  let whitney = WhitneyComplex::new(&topology, &metric);
  let dirac = HodgeDirac::assemble(&whitney);

  let dt = 0.02 * metric.mesh_width_min();
  let times: Vec<f64> = (0..=50).map(|i| dt * i as f64).collect();
  let initial = seed_field(&dirac);

  let implicit = solve_dirac(&whitney, &times, initial.clone());
  let explicit = solve_dirac_leapfrog(&whitney, &times, initial);

  let last_implicit = dirac.flatten(implicit.last().unwrap());
  let last_explicit = dirac.flatten(explicit.last().unwrap());
  let rel_err = (&last_implicit - &last_explicit).norm() / last_implicit.norm();
  assert!(rel_err < 1e-2, "solvers disagree by {rel_err}");
}
