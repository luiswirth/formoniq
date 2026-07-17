//! Module for the Wave Equation, the prototypical hyperbolic PDE.

use crate::{
  problems::elliptic::HodgeBlocks,
  time::{LinearIrk, Tableau},
  whitney_complex::HilbertComplex,
};

use common::linalg::{
  faer::FaerCholesky,
  nalgebra::{quadratic_form_sparse, CooMatrix, CooMatrixExt, CsrMatrix, Vector},
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

  /// The full Hodge wave energy
  /// $E = 1/2 (norm(delta u)^2 + norm(dif u)^2 + norm(diff_t u)^2)
  ///    = 1/2 (sigma^T M_sigma sigma + u^T K u + w^T M w)$, assembled from the
  /// mixed [`HodgeBlocks`] of the `complex` at `grade` --- the invariant
  /// [`solve_wave`] conserves. The down-part $norm(delta u)^2$ needs the
  /// codifferential $sigma = delta u$, recovered by the same $M_sigma$ solve
  /// the solver uses.
  pub fn energy<C: HilbertComplex>(&self, complex: &C, grade: ExteriorGrade) -> f64 {
    let hb = HodgeBlocks::compute(complex, grade);
    let inclusion = complex.inclusion(grade);
    let u = inclusion.transpose() * &self.pos;
    let w = inclusion.transpose() * &self.vel;
    let sigma_energy = if hb.n_sigma > 0 {
      let sigma = FaerCholesky::new(hb.mass_sigma.clone()).solve(&(hb.codif_dn() * &u));
      quadratic_form_sparse(&hb.mass_sigma, &sigma)
    } else {
      0.0
    };
    0.5
      * (sigma_energy
        + quadratic_form_sparse(&hb.stiff(), &u)
        + quadratic_form_sparse(&hb.mass_u, &w))
  }
}

/// Gauss-Legendre for the Hodge wave equation $diff_(t t) u = -Delta u + f$ on
/// Whitney $k$-forms of any `grade`, with the full Hodge Laplacian
/// $Delta = dif delta + delta dif$.
///
/// The down-part enters through the mixed auxiliary $sigma = delta u in
/// Lambda^(k-1)$, algebraic (no $diff_t sigma$). Recast first-order in
/// $y = (sigma, u, w)$ with $w = diff_t u$:
///
/// $ mat(0,0,0; 0,M,0; 0,0,M) dot(y) = mat(-M_sigma, C_"dn", 0; 0, 0, M;
///   -M D^(k-1), -K, 0) y + vec(0, 0, M f). $
///
/// The singular block mass makes this an index-1 DAE, but since the constraint
/// is linear the reduced $(u, w)$ dynamics are a genuine linear Hamiltonian
/// system with the full $Delta$, whose quadratic energy
/// $ E = 1/2 (norm(delta u)^2 + norm(dif u)^2 + norm(diff_t u)^2)
///     = 1/2 (sigma^T M_sigma sigma + u^T K u + w^T M w) $
/// Gauss-Legendre conserves *exactly* --- to roundoff, not merely bounded. This
/// is the same conserved energy as the three-field $(sigma, mu, omega)$ Hodge
/// wave system; the linear constraint makes the two algebraically equivalent,
/// and the $(sigma, u, w)$ form is chosen here because it carries $u$ itself.
///
/// `times` is assumed evenly spaced (the stage system is factored once).
/// Boundary conditions come entirely from the `complex` (natural on the full
/// [`WhitneyComplex`], homogeneous essential on the relative one); `initial`
/// and `force` are ambient, restricted internally and the returned states
/// extended back.
///
/// [`WhitneyComplex`]: crate::whitney_complex::WhitneyComplex
pub fn solve_wave<C: HilbertComplex>(
  complex: &C,
  grade: ExteriorGrade,
  times: &[f64],
  initial: WaveState,
  force_data: Cochain,
) -> Vec<WaveState> {
  let hb = HodgeBlocks::compute(complex, grade);
  let (ns, nu) = (hb.n_sigma, hb.n_u);

  let coo = CooMatrix::from;
  let z = CooMatrix::zeros;
  let mass_u = coo(&hb.mass_u);
  let mass_block = CsrMatrix::from(&CooMatrix::block(&[
    &[&z(ns, ns), &z(ns, nu), &z(ns, nu)],
    &[&z(nu, ns), &mass_u, &z(nu, nu)],
    &[&z(nu, ns), &z(nu, nu), &mass_u],
  ]));
  let op_block = CsrMatrix::from(&CooMatrix::block(&[
    &[&coo(&(-&hb.mass_sigma)), &coo(&hb.codif_dn()), &z(ns, nu)],
    &[&z(nu, ns), &z(nu, nu), &mass_u],
    &[&coo(&(-&hb.dif_sigma())), &coo(&(-&hb.stiff())), &z(nu, nu)],
  ]));

  let inclusion = complex.inclusion(grade);
  let u0 = inclusion.transpose() * initial.pos;
  let w0 = inclusion.transpose() * initial.vel;
  let sigma0 = if ns > 0 {
    FaerCholesky::new(hb.mass_sigma.clone()).solve(&(hb.codif_dn() * &u0))
  } else {
    Vector::zeros(0)
  };

  let force = &hb.mass_u * (inclusion.transpose() * force_data.coeffs());
  let mut forcing = Vector::zeros(ns + 2 * nu);
  forcing.rows_mut(ns + nu, nu).copy_from(&force);

  let dt = times.windows(2).next().map_or(0.0, |w| w[1] - w[0]);
  let irk = LinearIrk::new(Tableau::gauss_legendre(2), &mass_block, op_block, dt);

  let mut y = Vector::zeros(ns + 2 * nu);
  y.rows_mut(0, ns).copy_from(&sigma0);
  y.rows_mut(ns, nu).copy_from(&u0);
  y.rows_mut(ns + nu, nu).copy_from(&w0);

  let reconstruct = |y: &Vector| WaveState {
    pos: &inclusion * y.rows(ns, nu),
    vel: &inclusion * y.rows(ns + nu, nu),
  };

  let mut solution = Vec::with_capacity(times.len());
  solution.push(reconstruct(&y));
  for t01 in times.windows(2) {
    let [t0, _t1] = t01 else { unreachable!() };
    y = irk.step(&y, *t0, |_| forcing.clone());
    solution.push(reconstruct(&y));
  }

  solution
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

#[cfg(test)]
mod test {
  use super::*;
  use crate::whitney_complex::WhitneyComplex;
  use manifold::gen::cartesian::CartesianMeshInfo;

  use approx::assert_relative_eq;

  /// The hyperbolic law, at every dimension and grade: the full Hodge wave
  /// energy is conserved to roundoff. Gauss-Legendre is symplectic and, on this
  /// linear system, conserves the quadratic invariant exactly --- through the
  /// algebraic $sigma$ constraint and across the degenerate grades ($k = 0$: no
  /// $sigma$; $k = n$: $dif u = 0$, energy is pure kinetic).
  #[test]
  fn energy_conserved_at_every_grade() {
    for dim in 2..=3 {
      let (topology, coords) = CartesianMeshInfo::new_unit(dim, 2).compute_coord_complex();
      let metric = coords.to_edge_lengths(&topology);
      let whitney = WhitneyComplex::new(&topology, &metric);

      for grade in 0..=dim {
        let n = whitney.ndofs(grade);
        let pos = Vector::from_fn(n, |i, _| ((7 * i + 3) % 11) as f64 - 5.0);
        let vel = Vector::from_fn(n, |i, _| ((4 * i + 1) % 9) as f64 - 4.0);
        let force = Cochain::new(grade, Vector::zeros(n));

        let times: Vec<f64> = (0..=100).map(|i| 0.1 * i as f64).collect();
        let sol = solve_wave(&whitney, grade, &times, WaveState::new(pos, vel), force);

        let energy0 = sol[0].energy(&whitney, grade);
        for state in &sol {
          let energy = state.energy(&whitney, grade);
          assert_relative_eq!(energy, energy0, epsilon = 1e-8 * energy0.max(1.0));
        }
      }
    }
  }
}
