//! Maxwell's equations on Minkowski spacetime, covariantly: the source problem
//!
//! $ dif F = 0, quad delta F = J, $
//!
//! for the electromagnetic field 2-form $F in Lambda^2$ and the current
//! $J in Lambda^1$, on a simplicial mesh of the spacetime box
//! $[0, T] times [0, 1]^d$ carrying the Minkowski metric
//! $eta = "diag"(-1, +1, dots.c, +1)$. Time is one of the mesh directions;
//! there is no time-stepping loop, the hyperbolicity living entirely in the
//! signature.
//!
//! This is the *grade-2 sector* of the covariant Hodge--Dirac equation
//! $sans(D) F = J$, $sans(D) = dif + delta$, at zero mass. By grade,
//!
//! $ sans(D) F = underbrace(dif F, in Lambda^3) + underbrace(delta F, in Lambda^1)
//!   = underbrace(0, "grade 3") + underbrace(J, "grade 1"), $
//!
//! so the two Maxwell equations are two components of one equation: $dif F = 0$
//! (Bianchi, no magnetic monopoles) and $delta F = J$ (Gauss--Ampère). The field
//! is a single 2-form; the whole de Rham complex is carried because $sans(D)$ is
//! grade-mixing.
//!
//! # This example does not converge
//!
//! It is kept as the reproducer for the massless obstruction, not as a working
//! solve. The Galerkin posing below puts essential data on the *whole* spacetime
//! boundary, which for a hyperbolic operator is the Hadamard ill-posed problem:
//! the discrete systems are solved to a residual of $10^(-15)$ but their limit is
//! not the solution. In dimension 3 that stays hidden -- the error tracks the
//! interpolation floor -- and in dimension 4 the two columns separate. A causal
//! essential part and an $L^2$ least-squares formulation were both tried and
//! both fail, for reasons recorded in the issue.
//!
//! The obstruction is the inf-sup constant degenerating as $m -> 0$, not the
//! signature or the elements: the *massive* problem of `minkowski_dirac.rs`
//! converges at rate 1 in every dimension on the same machinery.
//!
//! See <https://github.com/luiswirth/formoniq/issues/117>.
//!
//! # Manufactured plane wave
//!
//! The exact field is an electromagnetic plane wave built from a null covector
//! $a$, $inner(a, a)_(eta^(-1)) = 0$ -- the massless dispersion relation
//! $sans(D)^2 F = square F = 0$. With a constant grade-1 polarization $omega$,
//!
//! $ F = sin(a dot x + phi) thin (a wedge omega), quad
//!   J = delta F = -cos(a dot x + phi) thin iota_(a^sharp)(a wedge omega). $
//!
//! Closedness is exact and structural: $dif F = cos(dot) thin a wedge a wedge
//! omega = 0$ because $a wedge a = 0$, so $F$ is a genuine Maxwell field rather
//! than one up to truncation. Errors are normed in the Euclidean comparison
//! geometry, since the indefinite Lorentzian pairing cannot norm one.

extern crate nalgebra as na;

use coorder::Coord;
use derham::{cochain::Cochain, project::derham_map, section::CoordFieldExt};
use exterior::MultiForm;
use formoniq::{
  assemble::assemble_galvec,
  fe::fe_l2_error,
  operators::SourceElVec,
  problems::dirac::{MixedField, solve_dirac_source},
  whitney_complex::WhitneyComplex,
};
use glatt::field::DiffFormClosure;
use gramian::Metric;
use simplicial::{
  atlas::SimplexQuadRule, geometry::coord::mesh::MeshCoords, linalg::Vector,
  mesher::cartesian::CartesianGrid, topology::ordering::CellOrdering,
};

use std::f64::consts::PI;

/// Phase offset of the plane wave, keeping it generic against the mesh and the
/// box periods (so no resonance mode is hit).
const PHASE: f64 = 0.3;

fn main() {
  for dim in [4, 3] {
    let nsubs: &[usize] = if dim == 3 { &[2, 4, 8] } else { &[1, 2, 4] };
    convergence(dim, nsubs);
  }
}

/// The null (lightlike) wave covector $a$ of the massless plane wave: the time
/// component is the spatial norm, $a_0 = norm(a_"space")$, so
/// $inner(a, a)_(eta^(-1)) = -a_0^2 + norm(a_"space")^2 = 0$. The spatial part is
/// generic (irrational against the box), giving a massless wave propagating
/// obliquely to the mesh axes.
fn wave_covector(dim: usize) -> MultiForm {
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
fn polarization(dim: usize) -> MultiForm {
  MultiForm::line(Vector::from_fn(dim, |i, _| 1.0 + 0.5 * (i as f64)))
}

fn convergence(dim: usize, nsubs: &[usize]) {
  let eta = Metric::minkowski(dim);
  let a = wave_covector(dim);
  let a_sharp = a.sharp(&eta);
  let a_vec = a.coeffs().clone();

  // The constant field bivector $Phi = a wedge omega$ and the current bivector
  // $-iota_(a^sharp) Phi$. Faraday: $dif F = cos(dot) thin a wedge Phi = 0$ since
  // $a wedge a = 0$, so $F$ is closed identically -- a genuine Maxwell field.
  let phi = a.wedge(&polarization(dim));
  let current = -1.0 * phi.interior_product(&a_sharp);

  // Null to roundoff; CausalType would read the residual sign, which says
  // nothing here.
  let null_norm_sq = a.inner(&a, &eta);
  println!(
    "Covariant Maxwell dif F = 0, delta F = J on [0,{:.1}] x [0,1]^{}, eta = diag(-1,+1,..): dim {dim}",
    simplicial::mesher::cartesian::CAUSAL_TIME_SCALE,
    dim - 1
  );
  println!("  null wave covector: <a,a>_eta^-1 = {null_norm_sq:+.2e} (massless dispersion)");
  println!(
    "  {:>5} | {:>9} | {:>10} | {:>6} | {:>10} | {:>11}",
    "nsub", "dofs", "rel error", "rate", "interp", "|dif F_h|"
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

    // $norm(F)_(L^2)$, as the error of the zero cochain: the scale that makes
    // the absolute error readable.
    let zero = Cochain::new(2, Vector::zeros(whitney.ndofs(2)));
    let scale = fe_l2_error(&zero, &field_section, &topology, &euclidean_lengths);
    // The interpolation error of the same field: the floor a Galerkin solution
    // is entitled to reach. The two columns separating is the obstruction.
    let interpolant = derham_map(&field_section, &topology, 3);
    let interp_error = fe_l2_error(&interpolant, &field_section, &topology, &euclidean_lengths);

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
    println!(
      "  {nsub:>5} | {ndofs:>9} | {:>10.3e} | {rate:>6} | {:>10.3e} | {closedness:>11.3e}",
      error / scale,
      interp_error / scale
    );
    previous = Some((nsub, error));
  }
  println!();
}
