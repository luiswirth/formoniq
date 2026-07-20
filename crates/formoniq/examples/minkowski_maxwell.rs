//! Maxwell's equations on Minkowski spacetime, covariantly: the source problem
//!
//! $ dif F = 0, quad delta F = J, $
//!
//! for the electromagnetic field 2-form $F in Lambda^2$ and the current
//! $J in Lambda^1$, solved on a simplicial mesh of the spacetime box
//! $[0, T] times [0, 1]^d$ carrying the Minkowski metric
//! $eta = "diag"(-1, +1, dots.c, +1)$. Time is one of the mesh directions;
//! there is no time-stepping loop, the hyperbolicity living entirely in the
//! signature.
//!
//! This is the *grade-2 sector* of the covariant Hodge--Dirac equation
//! $sans(D) F = J$, $sans(D) = dif + delta$, at zero mass. Writing it out by
//! grade,
//!
//! $ sans(D) F = underbrace(dif F, in Lambda^3) + underbrace(delta F, in Lambda^1)
//!   = underbrace(0, "grade 3") + underbrace(J, "grade 1"), $
//!
//! the two Maxwell equations are the grade-3 and grade-1 components of one
//! equation: $dif F = 0$ (no magnetic monopoles / Bianchi) and $delta F = J$
//! (Gauss--Ampère). The field is a single 2-form, but the Hodge--Dirac operator
//! is grade-mixing, so the whole de Rham complex is carried and the exact
//! solution is $F$ sitting in grade 2 alone, $(0, 0, F, 0, dots.c)$, which the
//! [`solve_dirac_source`] machinery of `minkowski_dirac.rs` solves unchanged --
//! same operator, the mass set to zero and the load restricted to a grade-1
//! current.
//!
//! # Massless well-posedness
//!
//! The massive Hodge--Dirac problem of `minkowski_dirac.rs` is invertible for
//! any nonzero $m$. Maxwell is the massless case, and massless is where the
//! all-boundary Dirichlet posing turns Fredholm: the continuous d'Alembertian
//! with data on the whole boundary can resonate (a box mode with matched
//! periods lies in the kernel). Two facts keep this run well-posed:
//!
//! - **The first-order operator has no gauge kernel.** Maxwell's gauge freedom
//!   $A |-> A + dif chi$ lives in the *potential*; solving for the field $F$
//!   directly there is nothing to fix. And $sans(D) = dif + delta$ is not the
//!   second-order $delta dif$ on the potential: its kernel is the *harmonic*
//!   fields $ker Delta$, not the closed ones, so the gauge modes never enter.
//!   No gauge fixing is needed because no gauge redundancy is being solved for.
//! - **Genericity dispatches the resonance.** On a Lorentzian mesh there is no
//!   Hodge theorem forcing a topological harmonic: every kernel mode is a
//!   resonance, a measure-zero coincidence of the wave covector, the box
//!   periods and the mesh. The null covector, the irrational time scale and a
//!   generic phase avoid it; the run reports the discrete solve's residual so
//!   invertibility is verified, not assumed.
//!
//! The physically honest alternative is a causal initial-value posing rather
//! than all-boundary Dirichlet; that is the separate 3+1 evolution path
//! (`examples/dirac.rs`), deliberately not taken here -- the point of this
//! example is the *covariant* formulation, spacetime as one geometric object.
//!
//! # Manufactured plane wave
//!
//! The exact field is a genuine electromagnetic plane wave built from a null
//! (lightlike) covector $a$, $inner(a, a)_(eta^(-1)) = 0$ -- the massless
//! dispersion relation $sans(D)^2 F = square F = 0$. With a constant grade-1
//! polarization $omega$ transverse to $a$, the field and current are
//!
//! $ F = sin(a dot x + phi) thin (a wedge omega), quad
//!   J = delta F = -cos(a dot x + phi) thin iota_(a^sharp)(a wedge omega). $
//!
//! Closedness is exact and structural: $dif F = cos(a dot x + phi) thin
//! a wedge a wedge omega = 0$ because $a wedge a = 0$, so $F$ is a *genuine*
//! Maxwell field ($dif F = 0$ identically), not merely one up to truncation. The
//! run measures the $L^2$ error of the recovered $F$ against this exact field in
//! the Euclidean comparison metric (the indefinite Lorentzian pairing cannot
//! norm an error), and separately the discrete closedness $norm(dif F_h)$ --
//! zero to interpolation on the lift, small on the solution.

extern crate nalgebra as na;

use coorder::Coord;
use derham::{cochain::Cochain, project::derham_map, section::CoordFieldExt};
use exterior::{Dim, MultiForm};
use formoniq::{
  assemble::assemble_galvec,
  fe::fe_l2_error,
  operators::SourceElVec,
  problems::dirac::{solve_dirac_source, MixedField},
  whitney_complex::WhitneyComplex,
};
use glatt::field::DiffFormClosure;
use gramian::{CausalType, Metric};
use simplicial::{
  atlas::SimplexQuadRule, gen::cartesian::CartesianGrid, geometry::coord::mesh::MeshCoords,
  linalg::Vector, topology::ordering::CellOrdering,
};

use std::f64::consts::PI;

/// Phase offset of the plane wave, keeping it generic against the mesh and the
/// box periods (so no resonance mode is hit).
const PHASE: f64 = 0.3;

fn main() {
  for dim in [4, 3] {
    let nsubs: &[usize] = match dim {
      2 => &[2, 4, 8, 16],
      3 => &[2, 4, 8],
      _ => &[1, 2, 4],
    };
    convergence(dim, nsubs);
  }
}

/// The null (lightlike) wave covector $a$ of the massless plane wave: the time
/// component is the spatial norm, $a_0 = norm(a_"space")$, so
/// $inner(a, a)_(eta^(-1)) = -a_0^2 + norm(a_"space")^2 = 0$. The spatial part is
/// generic (irrational against the box), giving a massless wave propagating
/// obliquely to the mesh axes.
fn wave_covector(dim: Dim) -> MultiForm {
  let space = [0.5, 0.3, 0.2];
  let space = &space[..dim - 1];
  let a0 = space.iter().map(|c| c * c).sum::<f64>().sqrt();
  let mut coeffs = Vec::with_capacity(dim);
  coeffs.push(a0);
  coeffs.extend_from_slice(space);
  MultiForm::line(PI * Vector::from_column_slice(&coeffs))
}

/// A constant grade-1 polarization $omega$, transverse-ish to $a$ so that
/// $a wedge omega != 0$ and the field is nontrivial. Deterministic and generic.
fn polarization(dim: Dim) -> MultiForm {
  MultiForm::line(Vector::from_fn(dim, |i, _| 1.0 + 0.5 * (i as f64)))
}

fn convergence(dim: Dim, nsubs: &[usize]) {
  let eta = Metric::minkowski(dim);
  let a = wave_covector(dim);
  let a_sharp = a.sharp(&eta);
  let a_vec = a.coeffs().clone();

  // The constant field bivector $Phi = a wedge omega$ and the current bivector
  // $-iota_(a^sharp) Phi$. Faraday: $dif F = cos(dot) thin a wedge Phi = 0$ since
  // $a wedge a = 0$, so $F$ is closed identically -- a genuine Maxwell field.
  let phi = a.wedge(&polarization(dim));
  let current = -1.0 * phi.interior_product(&a_sharp);

  let null_norm_sq = a.inner(&a, &eta);
  let null_char = CausalType::from_norm_sq(null_norm_sq);
  println!(
    "Covariant Maxwell dif F = 0, delta F = J on [0,{:.1}] x [0,1]^{}, eta = diag(-1,+1,..): dim {dim}",
    simplicial::gen::cartesian::CAUSAL_TIME_SCALE,
    dim - 1
  );
  println!(
    "  null wave covector: <a,a>_eta^-1 = {null_norm_sq:+.2e} ({null_char:?}) -- massless dispersion"
  );
  println!(
    "  {:>5} | {:>9} | {:>10} | {:>6} | {:>11}",
    "nsub", "dofs", "L2 error F", "rate", "|dif F_h|"
  );

  // One coarse causally generic Minkowski box, Freudenthal-refined into a tower;
  // the Minkowski ambient and the off-light-cone time scaling survive refinement.
  let (mut topology, mut coords) = CartesianGrid::minkowski(dim, 1);
  let mut ordering = CellOrdering::colex(&topology);
  let mut current_sub = 1usize;

  let mut previous: Option<(usize, f64)> = None;
  for &nsub in nsubs {
    assert_eq!(nsub % current_sub, 0, "a tower needs successive multiples");
    if nsub > current_sub {
      let sub = topology.refine_with(&ordering, nsub / current_sub);
      coords = coords.refine(&sub);
      ordering = sub.ordering().clone();
      topology = sub.into_complex();
      current_sub = nsub;
    }

    // `coords` carries the Minkowski ambient (the Lorentzian Regge data); its
    // Euclidean view is the positive comparison geometry errors are normed in.
    let euclidean = MeshCoords::new(coords.matrix().clone());
    let regge = coords.to_edge_lengths_sq(&topology);
    let euclidean_lengths = euclidean.to_edge_lengths_sq(&topology);

    let whitney = WhitneyComplex::new(&topology, &regge);
    let relative = whitney.relative();

    // The exact electromagnetic field $F = sin(a dot x + phi) Phi$, grade 2.
    let (a_f, phi_f) = (a_vec.clone(), phi.clone());
    let exact_field = DiffFormClosure::new(
      move |p: &Coord| (p.vector().dot(&a_f) + PHASE).sin() * phi_f.clone(),
      dim,
      2,
    );
    // The current $J = delta F = -cos(a dot x + phi) iota_(a^sharp) Phi$, grade 1.
    let (a_j, current_j) = (a_vec.clone(), current.clone());
    let source = DiffFormClosure::new(
      move |p: &Coord| (p.vector().dot(&a_j) + PHASE).cos() * current_j.clone(),
      dim,
      1,
    );

    // The essential data on the whole spacetime boundary: the exact mixed field,
    // $F$ in grade 2 and zero on every other grade. The load: the grade-1 current
    // functional, zero elsewhere ($dif F = 0$ leaves the grade-3 load empty).
    let lift = MixedField::new(
      (0..=dim)
        .map(|k| {
          if k == 2 {
            let section = exact_field.pullback_on(&topology, &euclidean);
            derham_map(&section, &topology, 3)
          } else {
            Cochain::new(k, Vector::zeros(whitney.ndofs(k)))
          }
        })
        .collect(),
    );
    let loads = MixedField::new(
      (0..=dim)
        .map(|k| {
          if k == 1 {
            let section = source.pullback_on(&topology, &euclidean);
            Cochain::new(
              1,
              assemble_galvec(
                &topology,
                &regge,
                SourceElVec::new(&section, Some(SimplexQuadRule::degree(dim, 3))),
              ),
            )
          } else {
            Cochain::new(k, Vector::zeros(whitney.ndofs(k)))
          }
        })
        .collect(),
    );

    // Massless Maxwell: the same self-adjoint Hodge-Dirac source solve, m = 0.
    let solution = solve_dirac_source(&relative, 0.0, &loads, &lift);
    let field = solution.grade(2);

    let field_section = exact_field.pullback_on(&topology, &euclidean);
    let error = fe_l2_error(field, &field_section, &topology, &euclidean_lengths);

    // $norm(F)_(L^2)$: the error of the zero cochain against the exact field,
    // the scale the absolute error is only meaningful against.
    let zero = Cochain::new(2, Vector::zeros(whitney.ndofs(2)));
    let scale = fe_l2_error(&zero, &field_section, &topology, &euclidean_lengths);
    // The interpolation error of the exact field: the floor any Galerkin
    // solution is measured against, separating solve error from element order.
    let interpolant = derham_map(&field_section, &topology, 3);
    let interp_error = fe_l2_error(&interpolant, &field_section, &topology, &euclidean_lengths);
    println!(
      "        (|F| = {scale:.3e}, rel = {:.3e}, interp = {:.3e})",
      error / scale,
      interp_error / scale
    );

    // Discrete closedness: the exact coboundary of the recovered field, which is
    // $R(dif F) = dif R(F) = 0$ exactly on the lift and small on the solution.
    let dif_field = &whitney.dif(2) * field.coeffs();
    let closedness = dif_field.norm() / field.coeffs().norm().max(1.0);

    if nsub == nsubs[0] {
      println!("  regge edge census: {}", regge.causal_census(&topology));
    }

    let ndofs: usize = (0..=dim).map(|k| whitney.ndofs(k)).sum();
    let rate = previous
      .map(|(n, e)| (e / error).ln() / (nsub as f64 / n as f64).ln())
      .map_or("--".into(), |r: f64| format!("{r:.2}"));
    println!("  {nsub:>5} | {ndofs:>9} | {error:>10.3e} | {rate:>6} | {closedness:>11.3e}");
    previous = Some((nsub, error));
  }
  println!();
}
