//! Module for the Wave Equation, the prototypical hyperbolic PDE.

use crate::{
  assemble, fe,
  linalg::quadratic_form_sparse,
  lse,
  mesh::SimplicialManifold,
  space::{DofIdx, FeSpace},
  util::FaerCholesky,
};

use std::rc::Rc;

pub struct WaveState {
  pub pos: na::DVector<f64>,
  pub vel: na::DVector<f64>,
}
impl WaveState {
  pub fn new(pos: na::DVector<f64>, vel: na::DVector<f64>) -> Self {
    Self { pos, vel }
  }

  /// The energy should be conserved from state to state.
  pub fn energy(&self, laplace: &nas::CscMatrix<f64>, mass: &nas::CscMatrix<f64>) -> f64 {
    0.5 * (quadratic_form_sparse(laplace, &self.pos) + quadratic_form_sparse(mass, &self.vel))
  }
}

/// times = [t_0,t_1,...,T]
pub fn solve_wave<F>(
  mesh: &Rc<SimplicialManifold>,
  times: &[f64],
  boundary_data: F,
  initial_data: WaveState,
  force_data: na::DVector<f64>,
) -> Vec<WaveState>
where
  F: Fn(DofIdx) -> f64,
{
  let space = FeSpace::new(mesh.clone());

  let mut laplace = assemble::assemble_galmat(&space, fe::laplace_beltrami_elmat);
  let mut mass = assemble::assemble_galmat(&space, fe::mass_elmat);
  let mut force = assemble::assemble_galvec(&space, fe::LoadElvec::new(force_data));

  lse::enforce_dirichlet_bc(mesh, &boundary_data, &mut laplace, &mut force);
  lse::enforce_dirichlet_bc(mesh, &boundary_data, &mut mass, &mut force);

  let laplace = laplace.to_nalgebra_csc();
  let mass = mass.to_nalgebra_csc();

  let mass_cholesky = FaerCholesky::new(mass.clone());

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
    let next = solve_wave_step(prev, dt, &force, &laplace, &mass, &mass_cholesky);

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
  forcing: &na::DVector<f64>,
  laplace: &nas::CscMatrix<f64>,
  mass: &nas::CscMatrix<f64>,
  mass_cholesky: &FaerCholesky,
) -> WaveState {
  let WaveState { pos, vel } = state;

  //let vel_half_step = mass_cholesky.solve(&(mass * vel + 0.5 * dt * (forcing - laplace * pos)));
  //let pos = pos + dt * &vel_half_step;
  //let vel = mass_cholesky.solve(&(mass * vel_half_step + 0.5 * dt * (forcing - laplace * pos)));

  let rhs = mass * vel + dt * (forcing - laplace * pos);
  let vel = mass_cholesky.solve(&rhs);
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
pub fn cfl_dt(mesh: &SimplicialManifold, vel: f64) -> f64 {
  const MARGIN: f64 = 0.95;
  MARGIN * mesh.mesh_width_min() / vel
}
