//! The massive Hodge–Dirac equation on Minkowski spacetime: the covariant
//! source problem
//!
//! $ (sans(D) + m) u = J, quad sans(D) = dif + delta, $
//!
//! solved on a simplicial mesh of the spacetime box $[0, T] times [0, 1]^d$
//! carrying the Minkowski metric $eta = "diag"(-1, +1, dots.c, +1)$ -- time is
//! one of the mesh directions, and the hyperbolicity of the equation lives
//! entirely in the signature of the metric, not in a time-stepping loop. The
//! flagship case is $1 + 3$ dimensional spacetime; the same code runs the
//! $1 + 1$ and $1 + 2$ cases, swept here for the convergence study.
//!
//! Three structures are on display, each printed by the run:
//!
//! - **The Lorentzian Hodge star.** On 2-forms in 4D, $star star = -1$ (versus
//!   $+1$ Riemannian): the star is a complex structure on $Lambda^2$, the
//!   electric-magnetic duality rotation. The table of $star (dif x^i wedge
//!   dif x^j)$ is checked against the closed form -- each timelike factor
//!   flips one sign.
//!
//! - **The Clifford structure.** The Hodge--Dirac operator acts on a plane
//!   wave $u = sin(a dot x) thin omega$ through the Clifford action of the
//!   wave covector, $(dif + delta) u = cos(a dot x) thin c_a omega$ with
//!   $c_a omega = a wedge omega - iota_(a^sharp) omega$, and
//!   $c_a c_a = -inner(a, a)_(eta^(-1))$: the dispersion relation
//!   $sans(D)^2 = Delta = square$ is literally the Clifford relation, with
//!   null covectors giving massless waves. The manufactured source $J$ below
//!   is *built* with this action, wedge and interior product supplying the
//!   two halves.
//!
//! - **The signature is the whole difference.** The discrete operator is the
//!   same `HodgeDirac` block assembly as the Riemannian 3+1 Maxwell evolution
//!   (`examples/dirac.rs`), in its self-adjoint sign $A = A^T$; the Minkowski
//!   metric enters only through the signed edge lengths the mesh carries. One
//!   pseudo-Riemannian code path, no Lorentzian special case.
//!
//! - **Regge calculus, as Regge intended.** The geometry the assembly actually
//!   consumes is not the coordinates but the Regge data derived from them: one
//!   signed squared length per edge (positive spacelike, zero null, negative
//!   timelike), from which each cell's Lorentzian metric is reconstructed by
//!   polarization -- "general relativity without coordinates". The run prints
//!   the causal census of the edges; the coordinates are used only on the I/O
//!   side, to interpolate the exact solution and to measure errors.
//!
//! The manufactured solution is the plane wave $u_k = sin(a dot x + phi)
//! thin omega_k$ on every grade at once (the field is genuinely mixed-grade:
//! $sans(D)$ couples neighbouring grades), with $J = (sans(D) + m) u$
//! analytic. Essential boundary values are the trace of the interpolant of
//! $u$, imposed by affine lifting, and the reported error is the $L^2$
//! distance of the solution to the exact plane wave, measured in the
//! *Euclidean comparison metric* on the same mesh -- the Lorentzian $L^2$
//! pairing is indefinite and cannot norm an error.
//!
//! Two facts about the discretization are worth seeing in the output:
//!
//! - **The mesh must be causally generic.** The time axis is scaled so that no
//!   edge or diagonal of the mesh is lightlike: a null edge degenerates the
//!   indefinite $L^2$ pairing on Whitney 1-forms *exactly* (the unit-box mesh
//!   in 2D has its diagonals on the light cone, and its 1-form mass matrix is
//!   singular, rank-deficient by the number of null-diagonal directions).
//!   This is a well-posedness condition of spacetime FEEC with no Riemannian
//!   analogue.
//!
//! - **Dirichlet data on the whole spacetime boundary.** The manufactured
//!   problem prescribes the trace on all of $diff([0,T] times [0,1]^d)$ --
//!   final data included -- which for a hyperbolic operator is a Fredholm
//!   boundary condition, not a causal initial-value one: the continuous
//!   problem can resonate (the massless wave $sin(pi t\/T') sin(pi x)$ with
//!   matched box periods is in the kernel). The mass term $m$ and the generic
//!   time scale keep the discrete system invertible here; the observed
//!   convergence is that of a manufactured-solution verification, not a claim
//!   of uniform hyperbolic well-posedness.

extern crate nalgebra as na;

use coorder::Coord;
use derham::{cochain::Cochain, project::derham_map, section::CoordFieldExt};
use exterior::{exterior_bases, exterior_dim, Dim, MultiForm};
use formoniq::{
  assemble::assemble_galvec,
  fe::fe_l2_error,
  operators::SourceElVec,
  problems::dirac::{solve_dirac_source, MixedField},
  whitney_complex::WhitneyComplex,
};
use glatt::field::DiffFormClosure;
use gramian::{CausalType, Gramian, Metric};
use multiindex::Sign;
use simplicial::{
  atlas::SimplexQuadRule, gen::cartesian::CartesianGrid, geometry::coord::mesh::MeshCoords,
  linalg::Vector, topology::ordering::CellOrdering,
};

use std::f64::consts::PI;

/// The time axis of the unit box is scaled by this factor: irrationally away
/// from 1, so no mesh edge or diagonal is lightlike (causal genericity) and no
/// box period resonates with the spatial ones.
const TIME_SCALE: f64 = 0.7;
/// The mass term $m$ of the Hodge--Dirac operator.
const MASS: f64 = 1.5;
/// Phase offset of the plane wave, keeping it generic against the mesh.
const PHASE: f64 = 0.3;

fn main() {
  lorentzian_star_table();
  clifford_dispersion();
  for dim in [2, 3, 4] {
    let nsubs: &[usize] = match dim {
      2 => &[2, 4, 8, 16],
      3 => &[2, 4, 8],
      _ => &[1, 2, 4],
    };
    convergence(dim, nsubs);
  }
}

/// Names of the coordinate axes of spacetime, time first.
const AXES: [&str; 4] = ["t", "x", "y", "z"];

fn blade_name(blade: &multiindex::Combination) -> String {
  blade
    .iter()
    .map(|i| format!("d{}", AXES[i]))
    .collect::<Vec<_>>()
    .join("^")
}

/// The Hodge star of the six 2-form blades of 4D Minkowski space: the
/// electric-magnetic duality, with $star star = -1$ -- a complex structure,
/// where the Riemannian star of middle grade in 4D squares to $+1$.
fn lorentzian_star_table() {
  let dim = 4;
  let eta = Metric::minkowski(dim);

  println!("Lorentzian Hodge star on 2-forms of Minkowski R^(1,3) (mostly-plus):");
  for blade in exterior_bases(dim, 2) {
    let form = MultiForm::from_blade_signed(dim, Sign::Pos, blade);
    let star = form.hodge_star(&eta);
    let star_star = star.hodge_star(&eta);

    let (coeff, star_blade) = star
      .basis_iter()
      .find(|(c, _)| *c != 0.0)
      .expect("star of a blade is a blade");
    let sign = if coeff > 0.0 { '+' } else { '-' };
    let involution = star_star.coeffs()[blade.rank()];
    println!(
      "  *({:>5}) = {sign}{:<5}   ** = {involution:+.0}",
      blade_name(&blade),
      blade_name(&star_blade),
    );
  }
  println!();
}

/// The Clifford relation behind the Dirac operator, evaluated on the actual
/// wave covector of the convergence study: $c_a c_a = -inner(a, a)_(eta^(-1))$,
/// the dispersion relation of $sans(D)^2 = square$.
fn clifford_dispersion() {
  println!("Wave covector a and its causal character under eta (mostly-plus):");
  for dim in [2, 3, 4] {
    let eta = Metric::minkowski(dim);
    let a = wave_covector(dim);
    let norm_sq = a.inner(&a, &eta);
    let causal = CausalType::from_norm_sq(norm_sq);
    println!("  dim {dim}: <a,a> = {norm_sq:+.4} ({causal:?}) -- D^2 plane-wave symbol");
  }
  println!();
}

/// The wave covector $a$ of the manufactured plane wave: timelike, generic
/// against the mesh axes.
fn wave_covector(dim: Dim) -> MultiForm {
  let components = [0.9, 0.5, 0.3, 0.2];
  MultiForm::line(PI * Vector::from_column_slice(&components[..dim]))
}

/// One constant blade per grade: the polarization $omega_k$ of the plane wave,
/// deterministic and fully populated so every rung of the complex couples.
fn polarization(dim: Dim, grade: usize) -> MultiForm {
  MultiForm::new(
    Vector::from_fn(exterior_dim(dim, grade), |i, _| {
      ((3 * i + 2 * grade) % 5) as f64 / 2.0 - 1.0
    }),
    dim,
    grade,
  )
}

fn convergence(dim: Dim, nsubs: &[usize]) {
  let eta = Metric::minkowski(dim);
  let a = wave_covector(dim);
  let a_sharp = a.sharp(&eta);
  let a_vec = a.coeffs().clone();

  // The grade-k component of the Clifford action $c_a omega = a wedge omega -
  // iota_(a^sharp) omega$ on the mixed-grade polarization: the wedge raises
  // grade k-1, the contraction lowers grade k+1.
  let clifford_component = |k: usize| -> MultiForm {
    let mut component = MultiForm::zero(dim, k);
    if k >= 1 {
      component += a.wedge(&polarization(dim, k - 1));
    }
    if k < dim {
      component -= polarization(dim, k + 1).interior_product(&a_sharp);
    }
    component
  };

  println!(
    "Hodge-Dirac (D + m) u = J on [0,{TIME_SCALE}] x [0,1]^{}, eta = diag(-1,+1,..): dim {dim}, m = {MASS}",
    dim - 1
  );
  println!(
    "  {:>5} | {:>9} | {:>10} | {:>6} | {:>10} | {:>6}",
    "nsub", "dofs", "L2 error", "rate", "interp err", "rate"
  );

  // One coarse cube, Freudenthal-refined `nsub`-fold per level: the mesh-agnostic
  // path, and on a Kuhn cube identical to the grid the generator would build.
  let (mut topology, mut coords) = CartesianGrid::new_unit(dim, 1).triangulate();
  let mut ordering = CellOrdering::colex(&topology);
  // The resolutions are successive multiples, so the levels form a refinement
  // tower: each is reached from the previous by the quotient factor, carrying
  // the ordering the subdivision inherits. That is what makes the tower agree
  // with refining the coarse cube once by `nsub`.
  let mut current = 1usize;

  let mut previous: Option<(usize, f64, f64)> = None;
  for &nsub in nsubs {
    assert_eq!(
      nsub % current,
      0,
      "a tower needs each resolution to be a multiple of the last"
    );
    if nsub > current {
      let sub = topology.refine_with(&ordering, nsub / current);
      coords = coords.refine(&sub);
      ordering = sub.ordering().clone();
      topology = sub.into_complex();
      current = nsub;
    }
    let mut matrix = coords.clone().into_matrix();
    matrix.row_mut(0).scale_mut(TIME_SCALE);
    // The same vertex coordinates seen twice: once inducing the Lorentzian
    // Regge data below, and as the Euclidean comparison geometry errors are
    // measured in (the indefinite pairing cannot norm an error).
    let euclidean = MeshCoords::new(matrix.clone());
    let spacetime = MeshCoords::with_ambient(matrix, Gramian::minkowski(dim));
    // The assembly geometry: signed squared edge lengths, nothing else. From
    // here on the spacetime is a Regge manifold; the embedding is forgotten.
    let regge = spacetime.to_edge_lengths_sq(&topology);
    // The Euclidean comparison geometry as intrinsic edge lengths: a positive
    // metric to norm errors in, the indefinite pairing cannot.
    let euclidean_lengths = euclidean.to_edge_lengths_sq(&topology);

    let whitney = WhitneyComplex::new(&topology, &regge);
    let relative = whitney.relative();

    let mut loads = Vec::with_capacity(dim + 1);
    let mut lift = Vec::with_capacity(dim + 1);
    let mut exact_sections = Vec::with_capacity(dim + 1);
    for k in 0..=dim {
      let omega = polarization(dim, k);
      let cliff = clifford_component(k);

      let (a_exact, omega_exact) = (a_vec.clone(), omega.clone());
      let exact = DiffFormClosure::new(
        move |p: &Coord| (p.vector().dot(&a_exact) + PHASE).sin() * omega_exact.clone(),
        dim,
        k,
      );
      let a_source = a_vec.clone();
      let source = DiffFormClosure::new(
        move |p: &Coord| {
          let phase = p.vector().dot(&a_source) + PHASE;
          phase.cos() * cliff.clone() + MASS * phase.sin() * omega.clone()
        },
        dim,
        k,
      );

      let exact_section = exact.pullback_on(&topology, &euclidean);
      let source_section = source.pullback_on(&topology, &euclidean);

      lift.push(derham_map(&exact_section, &topology, 3));
      loads.push(Cochain::new(
        k,
        assemble_galvec(
          &topology,
          &regge,
          SourceElVec::new(&source_section, Some(SimplexQuadRule::degree(dim, 3))),
        ),
      ));
      exact_sections.push(exact);
    }

    let lift = MixedField::new(lift);
    let solution = solve_dirac_source(&relative, MASS, &MixedField::new(loads), &lift);

    let l2_error_of = |field: &MixedField| -> f64 {
      (0..=dim)
        .map(|k| {
          let section = exact_sections[k].pullback_on(&topology, &euclidean);
          fe_l2_error(field.grade(k), &section, &topology, &euclidean_lengths).powi(2)
        })
        .sum::<f64>()
        .sqrt()
    };
    let error = l2_error_of(&solution);
    let interp_error = l2_error_of(&lift);
    let ndofs: usize = (0..=dim).map(|k| whitney.ndofs(k)).sum();

    if nsub == nsubs[0] {
      let mut census = [0usize; 3];
      for edge in topology.edges().handle_iter() {
        use simplicial::geometry::metric::mesh::EdgeRefExt;
        match edge.causal_type(&regge) {
          gramian::CausalType::Timelike => census[0] += 1,
          gramian::CausalType::Null => census[1] += 1,
          gramian::CausalType::Spacelike => census[2] += 1,
        }
      }
      println!(
        "  regge edge census at nsub={nsub}: {} timelike, {} null, {} spacelike",
        census[0], census[1], census[2]
      );
    }

    let rates = previous
      .map(|(n, e, i)| {
        let h_ratio = (nsub as f64 / n as f64).ln();
        (
          (e / error).ln() / h_ratio,
          (i / interp_error).ln() / h_ratio,
        )
      })
      .map_or(("--".into(), "--".into()), |(re, ri): (f64, f64)| {
        (format!("{re:.2}"), format!("{ri:.2}"))
      });
    println!(
      "  {:>5} | {:>9} | {:>10.3e} | {:>6} | {:>10.3e} | {:>6}",
      nsub, ndofs, error, rates.0, interp_error, rates.1
    );
    previous = Some((nsub, error, interp_error));
  }
  println!();
}
