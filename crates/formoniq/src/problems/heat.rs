//! Module for the Heat Equation, the prototypical parabolic PDE.

use simplicial::linalg::{CooMatrix, CooMatrixExt, CsrMatrix, Vector};

use crate::{
  hodge::HodgeBlocks,
  time::{LinearIrk, Tableau},
  whitney_complex::HilbertComplex,
};

use derham::Cochain;
use multialgebra::ExteriorGrade;

/// Radau IIA for the Hodge heat equation $partial_t u = -Delta u + f$ on Whitney
/// $k$-forms of any `grade`, with the full Hodge Laplacian
/// $Delta = dif delta + delta dif$.
///
/// The down-part $dif delta$ is reached through the mixed auxiliary
/// $sigma = delta u in Lambda^(k-1)$, whose defining relation is algebraic
/// (no $partial_t sigma$): the semidiscrete system
///
/// $ mat(0, 0; 0, M) dot(vec(sigma, u)) = mat(-M_sigma, C_"dn"; -M D^(k-1), -K)
///   vec(sigma, u) + vec(0, M f) $
///
/// is an index-1 differential-algebraic system, $M_sigma sigma = C_"dn" u$
/// slaves $sigma$ to $u$, and the singular block mass is exactly what encodes
/// that. Radau IIA is stiffly accurate and L-stable, the correct integrator
/// for such a DAE: it enforces the constraint at every stage and damps the
/// stiff modes monotonically. Following Arnold & Chen (FEEC for parabolic
/// problems), the harmonic component evolves freely and no gauge is imposed.
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
  grade: impl Into<ExteriorGrade>,
  nsteps: usize,
  dt: f64,
  initial: &Cochain,
  source: &Cochain,
  diffusion_coeff: f64,
) -> Vec<Cochain> {
  let grade = grade.into();
  let hb = HodgeBlocks::compute(complex, grade);
  let (ns, nu) = (hb.n_sigma, hb.n_u);

  let coo = CooMatrix::from;
  let mass_block = CsrMatrix::from(&CooMatrix::block(&[
    &[&CooMatrix::zeros(ns, ns), &CooMatrix::zeros(ns, nu)],
    &[&CooMatrix::zeros(nu, ns), &coo(&hb.mass_u)],
  ]));
  let op_block = CsrMatrix::from(&CooMatrix::block(&[
    &[&coo(&(-&hb.mass_sigma)), &coo(&hb.dif_test)],
    &[
      &coo(&(-diffusion_coeff * &hb.dif_test.transpose())),
      &coo(&(-diffusion_coeff * &hb.dif_both)),
    ],
  ]));

  let inclusion = complex.inclusion(grade);
  let u0 = inclusion.transpose() * initial.coeffs();
  // The algebraic constraint $M_sigma sigma_0 = C_"dn" u_0$ pins a consistent
  // initial $sigma$. An inconsistent one would pollute the first stage RHS.
  let sigma0 = hb.codif(&u0);

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
