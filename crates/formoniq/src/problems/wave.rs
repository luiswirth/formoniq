//! Module for the Wave Equation, the prototypical hyperbolic PDE.

use crate::{
  time::{LinearIrk, Tableau},
  whitney_complex::WhitneyComplex,
};

use common::linalg::nalgebra::{quadratic_form_sparse, CooMatrix, CooMatrixExt, CsrMatrix, Vector};
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

  /// The co-located energy $E = 1/2 (x^T K x + v^T M v)$, $K$ the stiffness
  /// and $M$ the mass. Gauss-Legendre is symplectic and, being applied here to
  /// a *linear* system, exactly conserves this quadratic invariant -- to
  /// roundoff, with no staggering and no $O(dif t)$ residual to correct for.
  pub fn energy(&self, laplace: &CsrMatrix, mass: &CsrMatrix) -> f64 {
    0.5 * (quadratic_form_sparse(laplace, &self.pos) + quadratic_form_sparse(mass, &self.vel))
  }
}

/// Gauss-Legendre (implicit midpoint) for the wave equation
/// $diff_(t t) u = -Delta u$ on Whitney $k$-forms of any `grade`, over
/// `times` = $[t_0, t_1, ..., T]$ (assumed evenly spaced: the stage system is
/// factored once for the whole run).
///
/// On a mesh with boundary, the boundary is held fixed at the initial
/// position: the velocity is constrained to the relative complex
/// (homogeneous essential condition). Recast as the first-order block system
/// $y = (x, v_0)$ -- $x$ the full position, $v_0$ the *reduced* velocity on
/// the relative complex, related by the inclusion $E$ --
///
/// $ mat(I, 0; 0, E^T M E) dot(y) = mat(0, E; -E^T K, 0) y
///   + mat(0; E^T f), $
///
/// which [`LinearIrk`] solves directly: symplectic and, since this system is
/// linear, exactly energy-conserving to machine precision -- not merely
/// bounded, unlike the symplectic-Euler leapfrog this replaces.
pub fn solve_wave(
  whitney: &WhitneyComplex,
  grade: ExteriorGrade,
  times: &[f64],
  initial_data: WaveState,
  force_data: Cochain,
) -> Vec<WaveState> {
  let laplace = CsrMatrix::from(&whitney.codif_dif(grade));
  let mass = CsrMatrix::from(&whitney.mass(grade));
  let d = mass.nrows();

  let force = &mass * force_data.coeffs();

  // The reduced velocity lives on the relative complex on meshes with
  // boundary; on a closed mesh the inclusion is just the identity.
  let inclusion = if whitney.boundary().is_some() {
    whitney.relative().inclusion(grade)
  } else {
    identity(d)
  };
  let d_rel = inclusion.ncols();

  let mass_rel = inclusion.transpose() * &mass * &inclusion;
  let neg_stiff_rel = inclusion.transpose() * (-&laplace);

  let sys_mass = CsrMatrix::from(&CooMatrix::block(&[
    &[&CooMatrix::from(&identity(d)), &CooMatrix::new(d, d_rel)],
    &[&CooMatrix::new(d_rel, d), &CooMatrix::from(&mass_rel)],
  ]));
  let sys_op = CsrMatrix::from(&CooMatrix::block(&[
    &[&CooMatrix::new(d, d), &CooMatrix::from(&inclusion)],
    &[
      &CooMatrix::from(&neg_stiff_rel),
      &CooMatrix::new(d_rel, d_rel),
    ],
  ]));

  let dt = times.windows(2).next().map_or(0.0, |w| w[1] - w[0]);
  let irk = LinearIrk::new(Tableau::gauss_legendre(2), &sys_mass, sys_op, dt);
  let forcing_rel = &inclusion.transpose() * &force;

  let mut y = Vector::zeros(d + d_rel);
  y.rows_mut(0, d).copy_from(&initial_data.pos);
  y.rows_mut(d, d_rel)
    .copy_from(&(inclusion.transpose() * &initial_data.vel));

  let mut solution = Vec::with_capacity(times.len());
  solution.push(initial_data);

  const ENERGY_TOLERANCE: f64 = 1e-6;
  let initial_energy = solution[0].energy(&laplace, &mass);

  for t01 in times.windows(2) {
    let [t0, _t1] = t01 else { unreachable!() };

    y = irk.step(&y, *t0, |_| {
      let mut f = Vector::zeros(d + d_rel);
      f.rows_mut(d, d_rel).copy_from(&forcing_rel);
      f
    });

    let pos = y.rows(0, d).into_owned();
    let vel = &inclusion * y.rows(d, d_rel);
    let next = WaveState { pos, vel };

    // Gauss-Legendre conserves this invariant to roundoff on the linear,
    // unforced system; a genuine blowup (e.g. an ill-posed force) is the only
    // thing that should move it. At top grade $K = 0$ the energy is
    // identically zero and there is nothing to check.
    let energy = next.energy(&laplace, &mass);
    if initial_energy > 0.0 && (energy - initial_energy).abs() >= ENERGY_TOLERANCE * initial_energy
    {
      panic!("Blowup while solving wave equation.");
    }

    solution.push(next);
  }

  solution
}

fn identity(n: usize) -> CsrMatrix {
  let mut coo = CooMatrix::new(n, n);
  for i in 0..n {
    coo.push(i, i, 1.0);
  }
  CsrMatrix::from(&coo)
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
