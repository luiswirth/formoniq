//! Module for the Heat Equation, the prototypical parabolic PDE.

use crate::linalg::faer::FaerCholesky;
use simplicial::linalg::{CooMatrix, CooMatrixExt, CsrMatrix, Vector};

use crate::{
  problems::elliptic::HodgeBlocks,
  time::{LinearIrk, Tableau},
  whitney_complex::HilbertComplex,
};

use derham::cochain::Cochain;
use exterior::ExteriorGrade;

/// Radau IIA for the Hodge heat equation $diff_t u = -Delta u + f$ on Whitney
/// $k$-forms of any `grade`, with the full Hodge Laplacian
/// $Delta = dif delta + delta dif$.
///
/// The down-part $dif delta$ is reached through the mixed auxiliary
/// $sigma = delta u in Lambda^(k-1)$, whose defining relation is *algebraic*
/// (no $diff_t sigma$): the semidiscrete system
///
/// $ mat(0, 0; 0, M) dot(vec(sigma, u)) = mat(-M_sigma, C_"dn"; -M D^(k-1), -K)
///   vec(sigma, u) + vec(0, M f) $
///
/// is an index-1 differential-algebraic system --- $M_sigma sigma = C_"dn" u$
/// slaves $sigma$ to $u$, and the singular block mass is exactly what encodes
/// that. Radau IIA is stiffly accurate and L-stable, the correct integrator
/// for such a DAE: it enforces the constraint at every stage and damps the
/// stiff modes monotonically. Following Arnold & Chen (*FEEC for parabolic
/// problems*), the harmonic component evolves freely --- no gauge is imposed.
///
/// Boundary conditions come entirely from the `complex`: the full
/// [`WhitneyComplex`] gives natural (Neumann) conditions, the relative complex
/// homogeneous essential (Dirichlet) ones. `initial` and `source` are ambient
/// cochains, restricted to the complex internally and the returned $u$ extended
/// back, so the caller is oblivious to the boundary condition.
///
/// [`WhitneyComplex`]: crate::whitney_complex::WhitneyComplex
pub fn solve_heat<C: HilbertComplex>(
  complex: &C,
  grade: ExteriorGrade,
  nsteps: usize,
  dt: f64,
  initial: &Cochain,
  source: &Cochain,
  diffusion_coeff: f64,
) -> Vec<Cochain> {
  let hb = HodgeBlocks::compute(complex, grade);
  let (ns, nu) = (hb.n_sigma, hb.n_u);

  let coo = CooMatrix::from;
  let mass_block = CsrMatrix::from(&CooMatrix::block(&[
    &[&CooMatrix::zeros(ns, ns), &CooMatrix::zeros(ns, nu)],
    &[&CooMatrix::zeros(nu, ns), &coo(&hb.mass_u)],
  ]));
  let op_block = CsrMatrix::from(&CooMatrix::block(&[
    &[&coo(&(-&hb.mass_sigma)), &coo(&hb.codif_dn())],
    &[
      &coo(&(-diffusion_coeff * &hb.dif_sigma())),
      &coo(&(-diffusion_coeff * &hb.stiff())),
    ],
  ]));

  let inclusion = complex.inclusion(grade);
  let u0 = inclusion.transpose() * initial.coeffs();
  // The algebraic constraint $M_sigma sigma_0 = C_"dn" u_0$ pins a consistent
  // initial $sigma$; an inconsistent one would pollute the first stage RHS.
  let sigma0 = if ns > 0 {
    FaerCholesky::new(hb.mass_sigma.clone()).solve(&(hb.codif_dn() * &u0))
  } else {
    Vector::zeros(0)
  };

  let source_u = &hb.mass_u * (inclusion.transpose() * source.coeffs());
  let mut forcing = Vector::zeros(ns + nu);
  forcing.rows_mut(ns, nu).copy_from(&source_u);

  let irk = LinearIrk::new(Tableau::radau_iia(2), &mass_block, op_block, dt);

  let mut y = Vector::zeros(ns + nu);
  y.rows_mut(0, ns).copy_from(&sigma0);
  y.rows_mut(ns, nu).copy_from(&u0);

  let mut solution = Vec::with_capacity(nsteps + 1);
  solution.push(Cochain::new(grade, &inclusion * &u0));
  for istep in 0..nsteps {
    y = irk.step(&y, istep as f64 * dt, |_| forcing.clone());
    let u = &inclusion * y.rows(ns, nu);
    solution.push(Cochain::new(grade, u));
  }

  solution
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::linalg::quadratic_form_sparse;
  use crate::problems::elliptic::solve_source;
  use crate::whitney_complex::WhitneyComplex;
  use simplicial::gen::cartesian::CartesianGrid;

  use approx::assert_relative_eq;

  /// The parabolic law, at every dimension and grade: with no source the
  /// $L^2$ energy $norm(u)_M^2$ of the Hodge heat flow can only decrease.
  /// $Delta$ is symmetric positive semidefinite, so the semidiscrete flow is a
  /// contraction and Radau IIA, being L-stable, inherits it unconditionally.
  /// The sweep exercises the degenerate grades too --- $k = 0$ (no $sigma$) and
  /// $k = n$ (no $omega$, $Delta = 0$, energy exactly flat).
  #[test]
  fn energy_dissipates_at_every_grade() {
    for dim in 1..=3 {
      let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
      let metric = coords.to_edge_lengths_sq(&topology);
      let whitney = WhitneyComplex::new(&topology, &metric);

      for grade in 0..=dim {
        let mass = CsrMatrix::from(&whitney.mass(grade));
        let n = whitney.ndofs(grade);
        let u0 = Cochain::new(
          grade,
          Vector::from_fn(n, |i, _| ((5 * i + 2) % 7) as f64 - 3.0),
        );
        let source = Cochain::new(grade, Vector::zeros(n));

        let sol = solve_heat(&whitney, grade, 30, 0.05, &u0, &source, 1.0);

        let mut prev = f64::INFINITY;
        for u in &sol {
          let energy = quadratic_form_sparse(&mass, u.coeffs());
          assert!(
            energy <= prev + 1e-9,
            "energy must not increase (dim {dim}, grade {grade})"
          );
          prev = energy;
        }
      }
    }
  }

  /// The full Hodge Laplacian, not merely its up-part: the steady state of
  /// $dot(u) = -Delta u + f$ must solve $Delta u = f$, i.e. reproduce the
  /// independently assembled and factored static mixed Hodge-Laplace solution
  /// [`solve_source`]. Run at grade $1$ on a topologically
  /// trivial box (relative $b_1 = 0$, so the steady state is unique), where the
  /// down-part $dif delta$ is genuinely nonzero --- the two code paths agreeing
  /// pins it down.
  #[test]
  fn steady_state_matches_static_hodge_laplace() {
    let (topology, coords) = CartesianGrid::new_unit(2, 3).triangulate();
    let metric = coords.to_edge_lengths_sq(&topology);
    let whitney = WhitneyComplex::new(&topology, &metric);
    let relative = whitney.relative();
    let grade = 1;
    assert_eq!(relative.harmonic_dim(grade), 0);

    let n_rel = relative.ndofs(grade);
    let f_rel = Vector::from_fn(n_rel, |i, _| ((3 * i + 1) % 5) as f64 - 2.0);
    let inclusion = relative.inclusion(grade);
    let source = Cochain::new(grade, &inclusion * &f_rel);

    let mass_rel = CsrMatrix::from(&relative.mass(grade));
    let galvec = &inclusion * (&mass_rel * &f_rel);
    let (_sigma, u_static, _p) = solve_source(&relative, galvec, grade).expect("static solve");

    // Radau IIA's fixed point is exactly the steady state, so a large step
    // reaches it fast (and exactly, independent of dt).
    let zero = Cochain::new(grade, Vector::zeros(whitney.ndofs(grade)));
    let sol = solve_heat(&relative, grade, 200, 1.0, &zero, &source, 1.0);
    let u_final = sol.last().unwrap();

    assert_relative_eq!(u_final.coeffs(), u_static.coeffs(), epsilon = 1e-7);
  }
}
