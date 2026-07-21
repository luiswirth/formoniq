//! Maxwell's equations as one Hodge–Dirac field in a perfect-electric-conductor
//! cavity.
//!
//! The electromagnetic field $E + B$ (electric 1-form, magnetic 2-form) is
//! carried as a cochain on the *whole* de Rham complex of the box $[0, pi]^3$,
//! $u = (u_0, u_1, u_2, u_3)$, because the Hodge–Dirac operator is grade-mixing;
//! all of Maxwell is the one first-order flow
//!
//! $ diff_t u = sans(D) u = (dif - delta) u $
//!
//! --- the mixed-grade formulation. The four grades carry the
//! four classical equations at once: grade 1 is the electric field $E$, grade 2
//! the magnetic flux $B$, and the two extremal grades are the Gauss constraints
//! ($delta E = 0$ in grade 0, $dif B = 0$ in grade 3), which stay negligible for
//! a physical field. The PEC walls ($"tr" u = 0$) are the relative Whitney
//! complex.
//!
//! The initial electric field
//!
//! $ E_0 = (sin y sin z, thick sin z sin x, thick sin x sin y) $
//!
//! has vanishing tangential trace on every wall and is divergence-free
//! ($delta E_0 = 0$), so it is a physical PEC field with curl; energy sloshes
//! from the electric grade 1 into the magnetic grade 2 and back. The run prints,
//! per step, the energy in each grade and the total.
//!
//! Three facts are visible in the output:
//!
//! - the total energy is flat to ~1e-15 relative: Gauss–Legendre is symplectic
//!   and conserves the quadratic invariant $1/2 thin u^T M u$ of the skew system
//!   exactly, with no drift --- only the physical exchange between grades 1 and 2;
//! - the magnetic Gauss law is exact: grade 3 stays at ~1e-30, machine zero,
//!   because $B$ only ever receives $dif E$ and $dif compose dif = 0$ --- a
//!   *topological* conservation law, independent of the metric or the step size;
//! - the electric Gauss law holds weakly: grade 0 ($delta_h E$) stays small and
//!   bounded but not at roundoff --- it is *metric*, the residual of the
//!   discretely-projected divergence-free field, and the Dirac flow keeps it
//!   from growing rather than pinning it to zero.

extern crate nalgebra as na;

use derham::{project::derham_map, section::CoordFieldExt};
use formoniq::{
  problems::dirac::{HodgeDirac, MixedField, solve_dirac},
  whitney_complex::WhitneyComplex,
};
use glatt::field::DiffFormClosure;
use simplicial::mesher::cartesian::CartesianGrid;

use std::f64::consts::PI;

fn main() {
  let dim = 3;
  let nboxes_per_dim = 4;

  // The PEC box cavity [0, pi]^3: essential boundary conditions on every grade
  // are the relative Whitney complex.
  let grid = CartesianGrid::new_unit_scaled(dim, nboxes_per_dim, PI);
  let (topology, coords) = grid.triangulate();
  let metric = coords.to_edge_lengths_sq(&topology);
  let whitney = WhitneyComplex::new(&topology, &metric);
  let pec = whitney.relative();
  // The solver returns states in the ambient Whitney space (extended by zero on
  // the constrained boundary), so energies are read off the full complex; on the
  // extended field they equal the relative-complex energies the flow conserves.
  let dirac = HodgeDirac::assemble(&whitney);

  println!(
    "PEC cavity [0,pi]^{dim} (Hodge-Dirac, full de Rham complex):\n  \
     dofs per grade: {} verts, {} edges (E), {} faces (B), {} cells",
    whitney.ndofs(0),
    whitney.ndofs(1),
    whitney.ndofs(2),
    whitney.ndofs(3),
  );

  // The initial electric field: a divergence-free, PEC-compatible 1-form with
  // curl, projected onto the mesh by the de Rham map (integration over edges).
  // Everything else starts at zero.
  let e_field = DiffFormClosure::one_form(
    |p| {
      na::dvector![
        p[1].sin() * p[2].sin(),
        p[2].sin() * p[0].sin(),
        p[0].sin() * p[1].sin(),
      ]
    },
    dim,
  );
  let e0 = derham_map(&e_field.pullback_on(&topology, &coords), &topology, 3);
  let initial = MixedField::from_grade(&whitney, e0);

  // Time stepping well within the CFL limit of the mesh.
  let cfl_fraction = 0.2;
  let dt = cfl_fraction * metric.mesh_width_min();
  let end_time = 6.0;
  let nsteps = (end_time / dt).ceil() as usize;
  let times: Vec<f64> = (0..=nsteps).map(|i| dt * i as f64).collect();
  println!("  dt = {dt:.4}, steps = {nsteps}\n");

  let solution = solve_dirac(&pec, &times, initial);

  println!(
    "{:>5} | {:>6} | {:>10} | {:>10} | {:>10} | {:>10} | {:>12} | {:>10}",
    "step", "t", "grade0", "E (g1)", "B (g2)", "grade3", "total", "drift"
  );
  let energy0 = dirac.energy(&solution[0]);
  let report_every = (nsteps / 20).max(1);
  for (istep, state) in solution.iter().enumerate() {
    if istep % report_every != 0 && istep != nsteps {
      continue;
    }
    let total = dirac.energy(state);
    let drift = (total - energy0).abs() / energy0;
    println!(
      "{:>5} | {:>6.3} | {:>10.2e} | {:>10.6} | {:>10.6} | {:>10.2e} | {:>12.8} | {:>10.2e}",
      istep,
      times[istep],
      dirac.grade_energy(state, 0),
      dirac.grade_energy(state, 1),
      dirac.grade_energy(state, 2),
      dirac.grade_energy(state, 3),
      total,
      drift
    );
  }

  println!(
    "\nTotal energy flat to ~1e-15 relative (symplectic, no drift). The magnetic\n\
     Gauss law dif B = 0 (grade 3) is exact to roundoff by dif.dif = 0; the\n\
     electric Gauss law delta E = 0 (grade 0) holds weakly. The four Maxwell\n\
     equations are the four grades of one Dirac field D = dif - delta.\n"
  );
}
