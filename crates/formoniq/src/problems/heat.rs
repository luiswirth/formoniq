//! Module for the Heat Equation, the prototypical parabolic PDE.

use common::linalg::nalgebra::{CsrMatrix, Vector};

use crate::{
  time::{LinearIrk, Tableau},
  whitney_complex::{BoundaryWhitneyComplex, WhitneyComplex},
};

use ddf::cochain::Cochain;
use exterior::ExteriorGrade;

/// Radau IIA (2-stage, order 3) for the heat equation $diff_t u = -Delta u$ on
/// Whitney $k$-forms of any `grade`, with the essential boundary condition
/// $"tr" u = g$ held fixed in time (affine lifting, factorized once).
///
/// Radau IIA is L-stable and algebraically stable: the structure that matters
/// here is not symplecticity (the heat flow has none) but monotone,
/// unconditionally stable damping of the diffusion operator's stiffest
/// eigenmodes -- exactly what an L-stable scheme, and not a merely A-stable
/// one such as Gauss-Legendre, guarantees.
///
/// The affine lift $u = hat(g) + E u_0$ ($hat(g)$ the zero-extension of `g`,
/// $E$ the inclusion of the relative complex) turns the constrained problem
/// into the unconstrained linear system $E^T M E dot(u_0) = E^T A E u_0 +
/// E^T (A hat(g) + "source")$ that [`LinearIrk`] solves directly.
#[allow(clippy::too_many_arguments)]
pub fn solve_heat(
  whitney: WhitneyComplex,
  boundary: &BoundaryWhitneyComplex,
  grade: ExteriorGrade,
  nsteps: usize,
  dt: f64,
  boundary_values: &Cochain,
  initial_data: Cochain,
  source_data: Cochain,
  diffusion_coeff: f64,
) -> Vec<Cochain> {
  let laplace = CsrMatrix::from(&whitney.codif_dif(grade));
  let mass = CsrMatrix::from(&whitney.mass(grade));
  let op = -diffusion_coeff * &laplace;

  let relative = whitney.relative();
  let inclusion = relative.inclusion(grade);
  let mass_rel = inclusion.transpose() * &mass * &inclusion;
  let op_rel = inclusion.transpose() * &op * &inclusion;

  let lift = boundary.extend_cochain(boundary_values).into_coeffs();
  let source = &mass * source_data.coeffs();
  let forcing_rel: Vector = inclusion.transpose() * (&op * &lift + &source);

  let irk = LinearIrk::new(Tableau::radau_iia(2), &mass_rel, op_rel, dt);

  let mut u0 = inclusion.transpose() * (initial_data.coeffs() - &lift);
  let mut solution = Vec::with_capacity(nsteps + 1);
  solution.push(initial_data);

  for istep in 0..nsteps {
    u0 = irk.step(&u0, istep as f64 * dt, |_| forcing_rel.clone());
    let u = &inclusion * &u0 + &lift;
    solution.push(Cochain::new(grade, u));
  }

  solution
}
