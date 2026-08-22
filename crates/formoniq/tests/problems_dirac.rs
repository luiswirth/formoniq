//! Laws for [`formoniq::problems::dirac`]: the defining Dirac identity
//! $sans(D)^2 = plus.minus Delta$, block by block and on either signature,
//! and the closed-form top-grade harmonic that the massless operator
//! annihilates.

use approx::assert_relative_eq;
use derham::Cochain;
use formoniq::{
  hodge::HodgeBlocks,
  linalg::faer::FaerCholesky,
  problems::dirac::{HodgeDirac, MixedField, top_harmonic},
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
