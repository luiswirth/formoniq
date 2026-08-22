//! The approximation theorems of the method: the Whitney space is first
//! order, so the Hodge-Laplace solution converges at $O(h)$ in $L^2$ and in
//! $H(dif)$, and the Galerkin solution is quasi-optimal, within a constant of
//! the best approximation the space admits.
//!
//! Everything else in the suite is exact algebra, topology or a conserved
//! quantity. This is the one place where the discrete object is compared
//! against the continuum one it approximates, so it is the only test that can
//! see an element matrix that is subtly wrong yet consistent with itself.

#[path = "../examples/util/mod.rs"]
mod util;

use {
  derham::{project::derham_map, section::CoordFieldExt},
  formoniq::{
    fe::{fe_l2_error, l2_projection},
    galerkin::LinearForm,
    operators::SourceForm,
    problems::elliptic,
    whitney_complex::WhitneyComplex,
  },
  regge::mesher::cartesian::CartesianGrid,
  simplicial::topology::ordering::CellOrdering,
  util::{BoundaryCondition, BoxEigenform, algebraic_convergence_rate},
};

use std::f64::consts::PI;

/// One refinement level of the source problem, read through the three maps
/// into the Whitney space: the Galerkin solution and the error of its
/// exterior derivative, the $L^2$ projection $P_h$ (the best approximation
/// the space admits), and the Whitney interpolation $W compose R$ of the
/// exact form (the one that commutes with $dif$).
struct Level {
  galerkin: f64,
  dif: Option<f64>,
  best: f64,
  interpolated: f64,
}

/// The source problem on the box $[0, pi]^n$ with the manufactured eigenform
/// as its exact solution, solved over a refinement tower.
///
/// The tower is a Freudenthal refinement of one coarse box, carrying the
/// ordering the subdivision inherits, rather than a grid regenerated per
/// level: the meshes are nested, so the coarse Whitney space is a subspace of
/// the fine one and every cell stays similar to the coarse box. Refining a
/// flat cell is exact, so no geometric error enters the rate.
fn tower(dim: usize, grade: usize, bc: BoundaryCondition, nlevels: usize) -> Vec<Level> {
  let form = BoxEigenform::new(dim, grade, bc);
  let (mut topology, mut coords) = CartesianGrid::new_unit_scaled(dim, 1, PI).triangulate();
  let mut lengths = coords.to_edge_lengths_sq(&topology);
  let mut ordering = CellOrdering::colex(&topology);

  (0..nlevels)
    .map(|level| {
      if level > 0 {
        let sub = topology.refine_with(&ordering, 2);
        lengths = lengths.refine(&sub, &topology);
        coords = coords.refine(&sub);
        ordering = sub.ordering().clone();
        topology = sub.into_complex();
      }
      let whitney = WhitneyComplex::new(&topology, &lengths);

      // The continuum eigenform becomes a field on the mesh by pullback along
      // the affine cell charts; everything downstream is intrinsic.
      let (exact, exact_load) = (form.solution(), form.load());
      let solution = exact.pullback_on(&topology, &coords);
      let load = exact_load.pullback_on(&topology, &coords);

      let source = SourceForm::new(&load, None).assemble(&topology, &lengths);
      let (_, galsol, _) = match bc {
        BoundaryCondition::Absolute => elliptic::solve_source(&whitney, source, grade).unwrap(),
        BoundaryCondition::Relative => {
          elliptic::solve_source(&whitney.relative(), source, grade).unwrap()
        }
      };

      let best = l2_projection(&solution, whitney, None);
      let interpolated = derham_map(&solution, &topology, 3);

      Level {
        galerkin: fe_l2_error(&galsol, &solution, &topology, &lengths),
        dif: form.dif_solution().map(|exact_dif| {
          let dif_solution = exact_dif.pullback_on(&topology, &coords);
          fe_l2_error(&galsol.dif(&topology), &dif_solution, &topology, &lengths)
        }),
        best: fe_l2_error(&best, &solution, &topology, &lengths),
        interpolated: fe_l2_error(&interpolated, &solution, &topology, &lengths),
      }
    })
    .collect()
}

/// The cases the theorems are stated over: every dimension, every grade, and
/// both boundary conditions, the two Hodge duals of the same problem.
fn cases() -> impl Iterator<Item = (usize, usize, BoundaryCondition)> {
  (1..=3).flat_map(|dim| {
    (0..=dim).flat_map(move |grade| {
      [BoundaryCondition::Absolute, BoundaryCondition::Relative]
        .into_iter()
        .map(move |bc| (dim, grade, bc))
    })
  })
}

/// Whitney forms are first order, so the discretization error is $O(h)$ in
/// $L^2 Lambda^k$ and in the $H(dif)$ seminorm, and the Galerkin solution is
/// quasi-optimal: Céa's lemma, in the form the inf-sup condition of the mixed
/// problem buys,
/// $norm(u - u_h) <= C inf_(v in cal(W) Lambda^k) norm(u - v)$
/// with $C$ independent of the mesh.
///
/// The two are one statement read at two levels: quasi-optimality says the
/// method inherits the approximation power of the space, and the rate is what
/// that power is. The infimum is bounded by the two computable approximants,
/// the $L^2$ projection and the Whitney interpolation of the exact form, so
/// the constant is measured rather than assumed. The interpolation carries a
/// rate of its own, which is Whitney's approximation theorem: it is the space
/// that is first order, and the method only inherits it.
///
/// The rate is read off the last halving of $h$, where it is asymptotic. A
/// rate this misses is not a slow method but a wrong one: an element matrix
/// that is wrong yet self-consistent converges to the wrong form at full
/// speed, or to the right one at half.
#[test]
fn the_solution_converges_at_the_whitney_rate_and_is_quasi_optimal() {
  const CONSTANT: f64 = 5.0;
  // First order is the claim; the threshold sits below $1$ because the
  // coarsest affordable levels in 3D are still pre-asymptotic, and a method
  // that is wrong rather than slow misses it by a factor, not a fraction.
  const RATE: f64 = 0.7;

  for (dim, grade, bc) in cases() {
    // One halving fewer in 3D: fixing a nontrivial harmonic part costs a
    // global solve there that grows far faster than the mesh does.
    let levels = tower(dim, grade, bc, if dim < 3 { 4 } else { 3 });
    let case = format!("dim {dim}, grade {grade}, {}", bc.label());

    for (level, l) in levels.iter().enumerate() {
      // Both the projection and the interpolation lie in the space, so the
      // infimum is under either: taking the smaller is the sharper reading of
      // the same inequality, and it does not depend on which of the two the
      // quadrature resolves better on a coarse mesh.
      let attainable = l.best.min(l.interpolated);
      assert!(
        l.galerkin <= CONSTANT * attainable + 1e-12,
        "{case}, level {level}: error {} against best approximation {attainable}",
        l.galerkin
      );
    }

    let [prev, last] = [levels.len() - 2, levels.len() - 1].map(|i| &levels[i]);
    for (name, prev, last) in [
      ("Galerkin", prev.galerkin, last.galerkin),
      ("interpolation", prev.interpolated, last.interpolated),
    ] {
      let rate = algebraic_convergence_rate(last, prev);
      assert!(rate > RATE, "{case}: {name} L2 rate {rate}");
    }

    if let (Some(prev), Some(last)) = (prev.dif, last.dif) {
      let hd = algebraic_convergence_rate(last, prev);
      assert!(hd > RATE, "{case}: H(dif) rate {hd}");
    }
  }
}
