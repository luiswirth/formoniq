//! Module for the Wave Equation, the prototypical hyperbolic PDE.

use crate::whitney_complex::WhitneyComplex;

use common::linalg::{
  faer::FaerCholesky,
  nalgebra::{bilinear_form_sparse, quadratic_form_sparse, CsrMatrix, Vector},
};
use ddf::cochain::Cochain;
use exterior::ExteriorGrade;
use manifold::geometry::metric::mesh::MeshLengths;

pub struct WaveState {
  pub pos: Vector,
  pub vel: Vector,
}
impl WaveState {
  pub fn new(pos: Vector, vel: Vector) -> Self {
    Self { pos, vel }
  }

  /// The exactly conserved leapfrog invariant
  /// $E = 1/2 (x^T K x + v^T M v) - (dif t)/2 x^T K v$, with $K$ the stiffness
  /// and $M$ the mass.
  ///
  /// The scheme is symplectic Euler, which stores position and velocity
  /// staggered by half a step; the co-located energy $1/2(x^T K x + v^T M v)$
  /// therefore oscillates at $O(dif t)$. The cross term is exactly the shadow
  /// correction that makes the quadratic form invariant under the unconstrained
  /// (linear) step map --- conserved to roundoff there, and to a bounded
  /// $O(dif t^2)$ once the essential velocity constraint projects each step.
  pub fn energy(&self, laplace: &CsrMatrix, mass: &CsrMatrix, dt: f64) -> f64 {
    0.5 * (quadratic_form_sparse(laplace, &self.pos) + quadratic_form_sparse(mass, &self.vel))
      - 0.5 * dt * bilinear_form_sparse(laplace, &self.pos, &self.vel)
  }
}

/// Leapfrog for the wave equation $diff_(t t) u = -Delta u$ on Whitney
/// $k$-forms of any `grade`, over `times` = $[t_0, t_1, ..., T]$.
///
/// On a mesh with boundary, the boundary is held fixed at the initial
/// position: the velocity is constrained to the relative complex
/// (homogeneous essential condition).
pub fn solve_wave(
  whitney: &WhitneyComplex,
  grade: ExteriorGrade,
  times: &[f64],
  initial_data: WaveState,
  force_data: Cochain,
) -> Vec<WaveState> {
  let laplace = CsrMatrix::from(&whitney.codif_dif(grade));
  let mass = CsrMatrix::from(&whitney.mass(grade));

  let force = &mass * force_data.coeffs();

  // Velocity mass matrix, constrained to the relative complex on meshes
  // with boundary: E (E^T M E)^-1 E^T.
  let inclusion = whitney
    .boundary()
    .is_some()
    .then(|| whitney.relative().inclusion(grade));
  let velocity_mass = match &inclusion {
    Some(incl) => incl.transpose() * &mass * incl,
    None => mass.clone(),
  };
  let mass_cholesky = FaerCholesky::new(velocity_mass);

  const ENERGY_TOLERANCE: f64 = 0.1; // 10%
  let dt0 = times.windows(2).next().map_or(0.0, |w| w[1] - w[0]);
  let initial_energy = initial_data.energy(&laplace, &mass, dt0);

  let mut solution = Vec::with_capacity(times.len());
  solution.push(initial_data);

  for t01 in times.windows(2) {
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

    // The conserved invariant stays flat (to $O(dif t^2)$ under the essential
    // constraint); a genuine blowup from an over-CFL step is the only thing that
    // moves it appreciably. At top grade $K = 0$ the energy is identically zero
    // and there is nothing to check.
    let energy = next.energy(&laplace, &mass, dt);
    if initial_energy > 0.0 && (energy - initial_energy).abs() >= ENERGY_TOLERANCE * initial_energy
    {
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
