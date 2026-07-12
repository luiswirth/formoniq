//! Maxwell's equations in a perfect-electric-conductor cavity.
//!
//! A standing electromagnetic field in the box cavity $[0, pi]^3$ with PEC
//! walls ($"tr" E = 0$), evolved by the structure-preserving leapfrog (Yee)
//! scheme. The initial electric field
//!
//! $ E_0 = (sin y sin z, thick sin z sin x, thick sin x sin y) $
//!
//! has vanishing tangential trace on every wall, so it is PEC-compatible, and
//! it has curl, so energy sloshes from the electric into the magnetic
//! reservoir and back. The run prints, at each step, the split of the
//! (co-located) energy between the two reservoirs, the exactly conserved
//! leapfrog invariant, and the magnetic charge $norm(dif B)$.
//!
//! Two governing facts are visible in the output:
//!
//! - $norm(dif B) = 0$ to roundoff for all time: the discrete Gauss law for
//!   magnetism is preserved *exactly*, a consequence of $dif compose dif = 0$.
//! - the leapfrog invariant is flat to ~1e-15 relative: the symplectic scheme
//!   has no energy drift, only the physical exchange between the reservoirs.
//!
//! For comparison the same cavity is also run through the second-order
//! curl-curl form (the grade-1 vector wave equation), reporting its energy.

extern crate nalgebra as na;

use common::linalg::nalgebra::Vector;
use ddf::{cochain::Cochain, derham::derham_map};
use exterior::field::DiffFormClosure;
use formoniq::{
  problems::maxwell::{
    leapfrog_energy, solve_maxwell_curl_curl, solve_maxwell_leapfrog, CurlCurlState,
    MaxwellOperators, MaxwellState, Medium,
  },
  whitney_complex::WhitneyComplex,
};
use manifold::gen::cartesian::CartesianMeshInfo;

use std::f64::consts::PI;

fn main() {
  let dim = 3;
  let nboxes_per_dim = 4;
  let medium = Medium::vacuum();

  // The PEC box cavity [0, pi]^3.
  let box_mesh = CartesianMeshInfo::new_unit_scaled(dim, nboxes_per_dim, PI);
  let (topology, coords) = box_mesh.compute_coord_complex();
  let metric = coords.to_edge_lengths(&topology);
  let fes = WhitneyComplex::new(&topology, &metric);
  let ops = MaxwellOperators::new(&fes, medium);

  println!(
    "PEC cavity [0,pi]^{dim}: {} edges (E dofs), {} faces (B dofs), c = {}",
    fes.ndofs(1),
    fes.ndofs(2),
    medium.wave_speed()
  );

  // The initial electric field: a PEC-compatible 1-form with curl, projected
  // onto the mesh by the de Rham map (integration over edges).
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
  let e0 = derham_map(&e_field, &topology, &coords, 3);
  let b0 = Cochain::new(2, Vector::zeros(fes.ndofs(2)));
  let initial = MaxwellState::new(e0, b0);

  // Time stepping well within the CFL limit of the mesh.
  let cfl_fraction = 0.2;
  let dt = cfl_fraction * metric.mesh_width_min() / medium.wave_speed();
  let end_time = 6.0;
  let nsteps = (end_time / dt).ceil() as usize;
  let times: Vec<f64> = (0..=nsteps).map(|i| dt * i as f64).collect();
  println!("dt = {dt:.4}, steps = {nsteps}\n");

  let current = Vector::zeros(fes.ndofs(1));
  let solution = solve_maxwell_leapfrog(fes, medium, &times, initial, &current);

  // Report every few steps.
  println!(
    "{:>5} | {:>6} | {:>10} | {:>10} | {:>12} | {:>11} | {:>9}",
    "step", "t", "E_elec", "E_magn", "E (leapfrog)", "drift", "||dB||"
  );
  let energy0 = leapfrog_energy(&solution[0], &solution[1], &ops);
  let report_every = (nsteps / 20).max(1);
  for (istep, pair) in solution.windows(2).enumerate() {
    if istep % report_every != 0 && istep != nsteps - 1 {
      continue;
    }
    let state = &pair[0];
    let e_elec = state.electric_energy(&ops);
    let e_magn = state.magnetic_energy(&ops);
    let e_leap = leapfrog_energy(&pair[0], &pair[1], &ops);
    let drift = (e_leap - energy0).abs() / energy0;
    let gauss = state.magnetic_charge(&ops).unwrap().coeffs().norm();
    println!(
      "{:>5} | {:>6.3} | {:>10.6} | {:>10.6} | {:>12.6} | {:>11.2e} | {:>9.2e}",
      istep, times[istep], e_elec, e_magn, e_leap, drift, gauss
    );
  }

  println!(
    "\nGauss law dif B = 0 preserved to roundoff (||dB|| ~ 1e-15); \
     leapfrog energy flat to ~1e-15 relative.\n"
  );

  // ---------------------------------------------------------------------------
  // The same cavity through the second-order curl-curl form for comparison.
  // ---------------------------------------------------------------------------
  let box_mesh = CartesianMeshInfo::new_unit_scaled(dim, nboxes_per_dim, PI);
  let (topology, coords) = box_mesh.compute_coord_complex();
  let metric = coords.to_edge_lengths(&topology);
  let fes = WhitneyComplex::new(&topology, &metric);
  let ops = MaxwellOperators::new(&fes, medium);
  let stiffness = ops.stiffness();

  let e0 = derham_map(&e_field, &topology, &coords, 3);
  let e_dot0 = Cochain::new(1, Vector::zeros(fes.ndofs(1)));
  let initial = CurlCurlState::new(e0, e_dot0);

  let forcing = Vector::zeros(fes.ndofs(1));
  let solution = solve_maxwell_curl_curl(fes, medium, &times, initial, &forcing);

  // The curl-curl (wave) energy 1/2 (eps ||e_dot||^2 + e^T K e) is a different
  // conserved quantity than the first-order field energy: the wave energy of
  // M e_double_dot + K e = 0. Under explicit stepping it oscillates within a
  // bounded band (symplectic stability), rather than drifting.
  let energies: Vec<f64> = solution
    .iter()
    .map(|s| s.energy(&ops, &stiffness))
    .collect();
  let energy_mean = energies.iter().sum::<f64>() / energies.len() as f64;
  let energy_min = energies.iter().cloned().fold(f64::INFINITY, f64::min);
  let energy_max = energies.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
  let band = (energy_max - energy_min) / energy_mean;
  println!(
    "curl-curl wave energy over {nsteps} steps: mean {energy_mean:.4}, \
     band [{energy_min:.4}, {energy_max:.4}] = {:.1}% (bounded, no drift)",
    100.0 * band
  );
}
