//! Module for the Wave Equation, the prototypical hyperbolic PDE.

use crate::whitney_complex::WhitneyComplex;

use common::linalg::{
  faer::FaerCholesky,
  nalgebra::{quadratic_form_sparse, CsrMatrix, Vector},
};
use ddf::cochain::Cochain;
use manifold::geometry::metric::mesh::MeshLengths;

pub struct WaveState {
  pub pos: Vector,
  pub vel: Vector,
}
impl WaveState {
  pub fn new(pos: Vector, vel: Vector) -> Self {
    Self { pos, vel }
  }

  /// The energy should be conserved from state to state.
  pub fn energy(&self, laplace: &CsrMatrix, mass: &CsrMatrix) -> f64 {
    0.5 * (quadratic_form_sparse(laplace, &self.pos) + quadratic_form_sparse(mass, &self.vel))
  }
}

/// times = [t_0,t_1,...,T]
///
/// On a mesh with boundary, the boundary is held fixed at the initial
/// position: the velocity is constrained to the relative complex
/// (homogeneous essential condition).
pub fn solve_wave(
  fes: WhitneyComplex,
  times: &[f64],
  initial_data: WaveState,
  force_data: Cochain,
) -> Vec<WaveState> {
  let laplace = CsrMatrix::from(&fes.codif_dif(0));
  let mass = CsrMatrix::from(&fes.mass(0));

  let force = &mass * force_data.coeffs();

  // Velocity mass matrix, constrained to the relative complex on meshes
  // with boundary: E (E^T M E)^-1 E^T.
  let inclusion = fes
    .boundary()
    .is_some()
    .then(|| fes.relative().inclusion(0));
  let velocity_mass = match &inclusion {
    Some(incl) => incl.transpose() * &mass * incl,
    None => mass.clone(),
  };
  let mass_cholesky = FaerCholesky::new(velocity_mass);

  const ENERGY_TOLERANCE: f64 = 0.1; // 10%
  let initial_energy = initial_data.energy(&laplace, &mass);

  let mut solution = Vec::with_capacity(times.len());
  solution.push(initial_data);

  let last_step = times.len() - 2;
  for (istep, t01) in times.windows(2).enumerate() {
    println!("Solving Wave Equation at step={istep}/{last_step}...");

    let [t0, t1] = t01 else { unreachable!() };
    let dt = t1 - t0;

    let prev = solution.last().unwrap();
    let next = solve_wave_step(
      prev,
      dt,
      &force,
      &laplace,
      &mass,
      &mass_cholesky,
      &inclusion,
    );

    let energy = next.energy(&laplace, &mass);
    if (energy - initial_energy).abs() >= ENERGY_TOLERANCE * initial_energy {
      panic!("Blowup while solving wave equation.");
    }

    solution.push(next);
  }

  solution
}

pub fn solve_wave_step(
  state: &WaveState,
  dt: f64,
  forcing: &Vector,
  laplace: &CsrMatrix,
  mass: &CsrMatrix,
  mass_cholesky: &FaerCholesky,
  inclusion: &Option<CsrMatrix>,
) -> WaveState {
  let WaveState { pos, vel } = state;

  let rhs = mass * vel + dt * (forcing - laplace * pos);
  let vel = match inclusion {
    Some(incl) => incl * mass_cholesky.solve(&(incl.transpose() * rhs)),
    None => mass_cholesky.solve(&rhs),
  };
  let pos = pos + dt * &vel;

  WaveState { pos, vel }
}

/// The optimal CFL time step for a simplicial mesh.
///
/// The Courant–Friedrichs–Lewy condition is the necessary condition C <= C_max
/// for the convergence of hyperbolic PDEs.
/// The Courant Number is defined as C = u dt/dx,
/// where u is the wave speed, dt is the time step and dx the minimal(!) length.
/// For explicit time stepping typically Cmax = 1.
/// Implicit time stepping is usually more lenient, allowing bigger values.
/// We assume here Cmax = 1, with a 5% safety margin.
pub fn cfl_dt(mesh_geo: &MeshLengths, vel: f64) -> f64 {
  const MARGIN: f64 = 0.95;
  MARGIN * mesh_geo.mesh_width_min() / vel
}
