//! Maxwell's equations on a simplicial Riemannian manifold.
//!
//! Electromagnetism is the canonical home of finite element exterior calculus:
//! the fields are differential forms and the field equations are the exterior
//! derivative on the de Rham complex. In the 3+1 split (space + time) on a
//! spatial Riemannian manifold $Omega$ the electromagnetic state is
//!
//! - the electric field $E in Lambda^1$, a *1-form* -- a 1-cochain of edge
//!   circulations (electromotive forces), and
//! - the magnetic flux $B in Lambda^2$, a *2-form* -- a 2-cochain of face
//!   fluxes.
//!
//! They sit on adjacent rungs of the de Rham complex
//! $Lambda^0 -> Lambda^1 -->^dif Lambda^2 -->^dif Lambda^3$, coupled by the
//! exterior derivative. The two constitutive relations
//! $D = epsilon star E$ and $H = mu^(-1) star B$ are the *Hodge stars*: in the
//! discretization they are exactly the Galerkin mass matrices
//! $M_1 = M_epsilon$ and $M_2 = M_(mu^(-1))$ of [`WhitneyComplex`].
//!
//! Maxwell's equations then read, with $D_1 = dif: C^1 -> C^2$ the discrete
//! curl and $D_1^T M_2$ the weak codifferential (discrete curl of $H$):
//!
//! $ dot(b) = -D_1 e             quad & "(Faraday's law, exact)" \
//!   M_epsilon dot(e) = D_1^T M_(mu^(-1)) b - j quad & "(Ampere's law, weak)" $
//!
//! Two structural facts make this a showcase for FEEC:
//!
//! - **Charge conservation is exact.** Because $D_2 D_1 = 0$ (the tested law
//!   $dif compose dif = 0$), the magnetic Gauss law $dif B = 0$ is preserved to
//!   roundoff by the [leapfrog integrator][solve_maxwell_leapfrog] -- no
//!   magnetic monopoles are ever created.
//! - **The wave equation is the scalar wave equation, one grade up.**
//!   Eliminating $b$ gives the curl-curl form
//!   $M_epsilon dot.double(e) + D_1^T M_(mu^(-1)) D_1 thin e = -dot(j)$, which is
//!   the grade-1 analogue of [`crate::problems::wave`]. The stiffness
//!   $D_1^T M_(mu^(-1)) D_1$ is the up-Laplacian [`WhitneyComplex::codif_dif`].
//!
//! Boundary conditions are the ones of the relative complex: a perfect
//! electric conductor (PEC) forces the tangential trace $"tr" E = 0$, i.e. the
//! electric field lives in the relative complex $C^1 (K, diff K)$ (see
//! [`crate::whitney_complex::RelativeWhitneyComplex`]); a perfect magnetic
//! conductor (PMC) is the natural do-nothing condition.
//!
//! The medium is homogeneous and isotropic, described by scalar $epsilon, mu$
//! ([`Medium`]); inhomogeneous or anisotropic media would weight the mass
//! matrices per cell.

use crate::whitney_complex::WhitneyComplex;

use common::linalg::{
  eigen::{sparse_shift_invert_eigen, EigenError},
  faer::FaerCholesky,
  nalgebra::{bilinear_form_sparse, quadratic_form_sparse, CsrMatrix, Vector},
};
use ddf::cochain::Cochain;

/// The constitutive parameters of a homogeneous isotropic linear medium: the
/// electric permittivity $epsilon$ and the magnetic permeability $mu$.
///
/// They enter only through the two constitutive Hodge stars, scaling the mass
/// matrices $M_epsilon = epsilon M_1$ and $M_(mu^(-1)) = mu^(-1) M_2$.
#[derive(Copy, Clone, Debug)]
pub struct Medium {
  pub epsilon: f64,
  pub mu: f64,
}
impl Medium {
  pub fn new(epsilon: f64, mu: f64) -> Self {
    assert!(
      epsilon > 0.0 && mu > 0.0,
      "epsilon and mu must be positive."
    );
    Self { epsilon, mu }
  }
  /// Vacuum in natural units: $epsilon = mu = 1$, so the wave speed $c = 1$.
  pub fn vacuum() -> Self {
    Self {
      epsilon: 1.0,
      mu: 1.0,
    }
  }
  /// The wave (light) speed $c = 1 \/ sqrt(epsilon mu)$.
  pub fn wave_speed(&self) -> f64 {
    1.0 / (self.epsilon * self.mu).sqrt()
  }
  /// The wave impedance $Z = sqrt(mu \/ epsilon)$.
  pub fn impedance(&self) -> f64 {
    (self.mu / self.epsilon).sqrt()
  }
}
impl Default for Medium {
  fn default() -> Self {
    Self::vacuum()
  }
}

/// The discrete Maxwell operators on the Whitney complex of a medium: the two
/// constitutive Hodge masses and the exterior derivative that couples the
/// electric 1-cochain to the magnetic 2-cochain.
///
/// Assembled once and shared by all the solvers below.
pub struct MaxwellOperators {
  /// $M_epsilon = epsilon M_1$: the electric mass, the discrete $epsilon star$.
  mass_e: CsrMatrix,
  /// $M_(mu^(-1)) = mu^(-1) M_2$: the magnetic mass, the discrete $mu^(-1) star$.
  mass_b: CsrMatrix,
  /// $D_1 = dif: C^1 -> C^2$: the discrete curl (Faraday's law).
  curl: CsrMatrix,
  /// $D_1^T M_(mu^(-1)): C^2 -> C^1$: the weak curl of $H$ (Ampere's law).
  weak_curl: CsrMatrix,
  /// $D_2 = dif: C^2 -> C^3$: the discrete divergence of the magnetic flux,
  /// present only when $n >= 3$ (a 3-form to be a magnetic charge).
  div_b: Option<CsrMatrix>,
}
impl MaxwellOperators {
  pub fn new(fes: &WhitneyComplex, medium: Medium) -> Self {
    assert!(
      fes.dim() >= 2,
      "Maxwell needs spatial dimension >= 2: E is a 1-form, B a 2-form."
    );
    let mass_e = medium.epsilon * &CsrMatrix::from(&fes.mass(1));
    let mass_b = (1.0 / medium.mu) * &CsrMatrix::from(&fes.mass(2));
    let curl = fes.dif(1);
    let weak_curl = curl.transpose() * &mass_b;
    let div_b = (fes.dim() >= 3).then(|| fes.dif(2));
    Self {
      mass_e,
      mass_b,
      curl,
      weak_curl,
      div_b,
    }
  }

  /// The curl-curl stiffness $K = D_1^T M_(mu^(-1)) D_1$, the up-Laplacian on
  /// 1-forms weighted by $mu^(-1)$.
  pub fn stiffness(&self) -> CsrMatrix {
    &self.weak_curl * &self.curl
  }
}

/// The electromagnetic state: the electric field as a 1-cochain and the
/// magnetic flux as a 2-cochain.
#[derive(Clone)]
pub struct MaxwellState {
  /// $e in C^1$: electric field 1-cochain (edge circulations).
  pub e: Cochain,
  /// $b in C^2$: magnetic flux 2-cochain (face fluxes).
  pub b: Cochain,
}
impl MaxwellState {
  pub fn new(e: Cochain, b: Cochain) -> Self {
    assert_eq!(e.grade(), 1, "The electric field is a 1-form.");
    assert_eq!(b.grade(), 2, "The magnetic flux is a 2-form.");
    Self { e, b }
  }

  /// The electric field energy $1/2 epsilon norm(E)^2$.
  pub fn electric_energy(&self, ops: &MaxwellOperators) -> f64 {
    0.5 * quadratic_form_sparse(&ops.mass_e, self.e.coeffs())
  }
  /// The magnetic field energy $1/2 mu^(-1) norm(B)^2$.
  pub fn magnetic_energy(&self, ops: &MaxwellOperators) -> f64 {
    0.5 * quadratic_form_sparse(&ops.mass_b, self.b.coeffs())
  }
  /// The electromagnetic energy
  /// $cal(E) = 1/2 (epsilon norm(E)^2 + mu^(-1) norm(B)^2)$,
  /// the sum of the electric and magnetic field energies. Co-located in time;
  /// see [`leapfrog_energy`] for the exactly conserved staggered invariant.
  pub fn energy(&self, ops: &MaxwellOperators) -> f64 {
    self.electric_energy(ops) + self.magnetic_energy(ops)
  }

  /// The magnetic charge $dif B$: a 3-cochain that vanishes identically for a
  /// physical field (Gauss's law for magnetism, no monopoles). The leapfrog
  /// integrator preserves it exactly, so it stays at its initial value for all
  /// time. `None` in dimension $n < 3$, where $dif B$ has no room to live.
  pub fn magnetic_charge(&self, ops: &MaxwellOperators) -> Option<Cochain> {
    ops
      .div_b
      .as_ref()
      .map(|div| Cochain::new(3, div * self.b.coeffs()))
  }
}

/// The exactly conserved discrete energy of the leapfrog scheme: the
/// time-staggered bilinear form
///
/// $ H^n = 1/2 epsilon (e^n)^T M_1 e^n
///        + 1/2 mu^(-1) (b^(n-1/2))^T M_2 b^(n+1/2). $
///
/// Unlike the co-located [`MaxwellState::energy`] -- which oscillates at
/// $O((omega dif t)^2)$ because $e$ and $b$ live at different times -- this
/// leapfrog invariant is conserved to roundoff in the source-free case (and is
/// positive definite precisely under the CFL condition). `curr` supplies $e^n$
/// and $b^(n-1/2)$, `next` supplies $b^(n+1/2)$: the two consecutive states of
/// the trajectory returned by [`solve_maxwell_leapfrog`].
pub fn leapfrog_energy(curr: &MaxwellState, next: &MaxwellState, ops: &MaxwellOperators) -> f64 {
  0.5
    * (quadratic_form_sparse(&ops.mass_e, curr.e.coeffs())
      + bilinear_form_sparse(&ops.mass_b, curr.b.coeffs(), next.b.coeffs()))
}

/// Solve Maxwell's equations by the leapfrog (Yee) scheme: the
/// structure-preserving first-order mixed integrator.
///
/// The electric 1-cochain $e^n$ lives at integer time steps, the magnetic
/// 2-cochain $b^(n+1/2)$ at the half steps, staggered as in Yee's scheme but
/// on the unstructured simplicial complex:
///
/// $ b^(n+1/2) &= b^(n-1/2) - dif t thin D_1 e^n
///   quad & "(Faraday, exact)" \
///   M_epsilon e^(n+1) &= M_epsilon e^n
///     + dif t (D_1^T M_(mu^(-1)) b^(n+1/2) - j)
///   quad & "(Ampere, weak)" $
///
/// Because $D_2 D_1 = 0$, the discrete Gauss law
/// $dif b^(n+1/2) = dif b^(n-1/2)$ holds to roundoff: no magnetic charge is
/// created. On a mesh with boundary the electric field is constrained to the
/// relative complex ($"tr" e = 0$): the perfect-electric-conductor (PEC)
/// condition. The initial electric field is projected onto the relative
/// complex to satisfy it.
///
/// `current` is the assembled current load $j_sigma = integral_Omega
/// angle.l J, W_sigma angle.r$ (a covector on the 1-cochains, e.g. from
/// [`crate::operators::SourceElVec`]), held constant in time; pass a zero
/// vector for the source-free case. `times` is the sequence
/// $[t_0, t_1, dots, T]$ of time nodes.
pub fn solve_maxwell_leapfrog(
  fes: WhitneyComplex,
  medium: Medium,
  times: &[f64],
  initial: MaxwellState,
  current: &Vector,
) -> Vec<MaxwellState> {
  let ops = MaxwellOperators::new(&fes, medium);

  // PEC: the electric field is constrained to the relative complex.
  let inclusion = fes
    .boundary()
    .is_some()
    .then(|| fes.relative().inclusion(1));

  // Electric mass, constrained to the relative complex on meshes with
  // boundary: solving Ampere's law is a mass solve, factorized once.
  let electric_mass = match &inclusion {
    Some(incl) => incl.transpose() * &ops.mass_e * incl,
    None => ops.mass_e.clone(),
  };
  let electric_cholesky = FaerCholesky::new(electric_mass);

  // Project the initial electric field onto the relative complex (PEC).
  let mut e = match &inclusion {
    Some(incl) => incl * (incl.transpose() * initial.e.coeffs()),
    None => initial.e.coeffs().clone(),
  };
  let mut b = initial.b.coeffs().clone();

  let mut solution = Vec::with_capacity(times.len());
  solution.push(MaxwellState::new(
    Cochain::new(1, e.clone()),
    Cochain::new(2, b.clone()),
  ));

  let last_step = times.len().saturating_sub(2);
  for (istep, t01) in times.windows(2).enumerate() {
    println!("Solving Maxwell (leapfrog) at step={istep}/{last_step}...");
    let [t0, t1] = t01 else { unreachable!() };
    let dt = t1 - t0;

    // Faraday: b advances by the exact discrete curl of e.
    b -= dt * (&ops.curl * &e);

    // Ampere: solve M_epsilon (e^(n+1) - e^n) = dt (D_1^T M_(mu^-1) b - j),
    // the electric-field increment, constrained to the relative complex.
    let rhs = dt * (&ops.weak_curl * &b - current);
    let delta = match &inclusion {
      Some(incl) => incl * electric_cholesky.solve(&(incl.transpose() * rhs)),
      None => electric_cholesky.solve(&rhs),
    };
    e += delta;

    solution.push(MaxwellState::new(
      Cochain::new(1, e.clone()),
      Cochain::new(2, b.clone()),
    ));
  }

  solution
}

/// The state of the second-order (curl-curl) wave form: the electric field and
/// its time derivative, both 1-cochains.
#[derive(Clone)]
pub struct CurlCurlState {
  /// $e in C^1$: electric field 1-cochain.
  pub e: Cochain,
  /// $dot(e) in C^1$: its time derivative.
  pub e_dot: Cochain,
}
impl CurlCurlState {
  pub fn new(e: Cochain, e_dot: Cochain) -> Self {
    assert_eq!(e.grade(), 1);
    assert_eq!(e_dot.grade(), 1);
    Self { e, e_dot }
  }

  /// The energy $cal(E) = 1/2 (epsilon norm(dot(e))^2 + e^T K e)$ of the
  /// curl-curl wave, conserved in the source-free case.
  pub fn energy(&self, ops: &MaxwellOperators, stiffness: &CsrMatrix) -> f64 {
    0.5
      * (quadratic_form_sparse(&ops.mass_e, self.e_dot.coeffs())
        + quadratic_form_sparse(stiffness, self.e.coeffs()))
  }
}

/// Solve the second-order curl-curl wave form of Maxwell for the electric
/// field alone,
///
/// $ M_epsilon dot.double(e) + K e = -dot(j),
///   quad K = D_1^T M_(mu^(-1)) D_1, $
///
/// the vector wave equation obtained by eliminating the magnetic field. This
/// is the grade-1 analogue of the scalar wave equation
/// [`crate::problems::wave`]; the stiffness $K$ is the up-Laplacian
/// [`WhitneyComplex::codif_dif`] weighted by $mu^(-1)$.
///
/// Explicit (symplectic Euler) time stepping, with the same PEC treatment as
/// [`solve_maxwell_leapfrog`]: $dot(e)$ is constrained to the relative complex.
/// `forcing` is the constant load $-dot(j)$; pass a zero vector for the
/// source-free case.
pub fn solve_maxwell_curl_curl(
  fes: WhitneyComplex,
  medium: Medium,
  times: &[f64],
  initial: CurlCurlState,
  forcing: &Vector,
) -> Vec<CurlCurlState> {
  let ops = MaxwellOperators::new(&fes, medium);
  let stiffness = ops.stiffness();

  let inclusion = fes
    .boundary()
    .is_some()
    .then(|| fes.relative().inclusion(1));
  let velocity_mass = match &inclusion {
    Some(incl) => incl.transpose() * &ops.mass_e * incl,
    None => ops.mass_e.clone(),
  };
  let mass_cholesky = FaerCholesky::new(velocity_mass);

  // Project the initial data onto the relative complex (PEC).
  let project = |v: &Vector| match &inclusion {
    Some(incl) => incl * (incl.transpose() * v),
    None => v.clone(),
  };
  let mut e = project(initial.e.coeffs());
  let mut e_dot = project(initial.e_dot.coeffs());

  let mut solution = Vec::with_capacity(times.len());
  solution.push(CurlCurlState::new(
    Cochain::new(1, e.clone()),
    Cochain::new(1, e_dot.clone()),
  ));

  let last_step = times.len().saturating_sub(2);
  for (istep, t01) in times.windows(2).enumerate() {
    println!("Solving Maxwell (curl-curl) at step={istep}/{last_step}...");
    let [t0, t1] = t01 else { unreachable!() };
    let dt = t1 - t0;

    // Symplectic Euler: update the velocity with the current stiffness force,
    // then advance the field with the new velocity.
    let rhs = &ops.mass_e * &e_dot + dt * (forcing - &stiffness * &e);
    e_dot = match &inclusion {
      Some(incl) => incl * mass_cholesky.solve(&(incl.transpose() * rhs)),
      None => mass_cholesky.solve(&rhs),
    };
    e += dt * &e_dot;

    solution.push(CurlCurlState::new(
      Cochain::new(1, e.clone()),
      Cochain::new(1, e_dot.clone()),
    ));
  }

  solution
}

/// Resonant modes of a perfect-electric-conductor cavity: the generalized
/// eigenvalue problem
///
/// $ K e = omega^2 M_epsilon e, quad K = D_1^T M_(mu^(-1)) D_1, $
///
/// on the relative complex $C^1 (K, diff K)$. The nonzero eigenvalues
/// $omega^2$ are the squared resonant angular frequencies of the cavity; the
/// kernel of $K$ (gradient fields and, on a topologically nontrivial domain,
/// harmonic fields) are the static zero modes. Returns the eigenvalues
/// $omega^2$ together with the corresponding electric-field eigen-cochains,
/// extended by zero from the relative complex onto the full mesh.
///
/// The frequency-domain counterpart of the time-domain solvers above, built
/// from the very same operators. Solved by [`sparse_shift_invert_eigen`] for
/// the eigenvalues nearest $0$. $ker K$ contains every discrete gradient
/// field, so `nmodes` must exceed its dimension before a resonance appears
/// among the modes returned.
pub fn solve_maxwell_cavity_modes(
  fes: WhitneyComplex,
  medium: Medium,
  nmodes: usize,
) -> Result<(Vector, Vec<Cochain>), EigenError> {
  let ops = MaxwellOperators::new(&fes, medium);
  let relative = fes.relative();
  let incl = relative.inclusion(1);

  let stiffness_rel = incl.transpose() * &ops.stiffness() * &incl;
  let mass_rel = incl.transpose() * &ops.mass_e * &incl;

  let (eigenvals, eigenvecs) = sparse_shift_invert_eigen(&stiffness_rel, &mass_rel, 0.0, nmodes)?;

  let modes = eigenvecs
    .column_iter()
    .map(|c| Cochain::new(1, &incl * c.into_owned()))
    .collect();

  Ok((eigenvals, modes))
}

#[cfg(test)]
mod test {
  use super::*;

  use common::linalg::nalgebra::Vector;
  use manifold::gen::cartesian::CartesianMeshInfo;

  use approx::assert_relative_eq;

  /// A deterministic, PEC-incompatible-but-projected initial electric field
  /// on the 1-cochains: enough curl to exercise the coupling.
  fn seed_electric_field(ndofs: usize) -> Vector {
    Vector::from_fn(ndofs, |i, _| ((3 * i + 1) % 7) as f64 - 3.0)
  }

  /// The crown jewel of the structure-preserving scheme: the discrete magnetic
  /// Gauss law $dif B = 0$ is preserved *exactly* (to roundoff) by the
  /// leapfrog integrator, because $dif compose dif = 0$. No magnetic monopoles
  /// are ever created, regardless of the field, the medium or the step size.
  #[test]
  fn magnetic_gauss_law_preserved_exactly() {
    let (topology, coords) = CartesianMeshInfo::new_unit(3, 2).compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);
    let fes = WhitneyComplex::new(&topology, &metric);
    let ops = MaxwellOperators::new(&fes, Medium::vacuum());

    let e0 = Cochain::new(1, seed_electric_field(fes.ndofs(1)));
    let b0 = Cochain::new(2, Vector::zeros(fes.ndofs(2)));
    let initial = MaxwellState::new(e0, b0);

    let times: Vec<f64> = (0..=20).map(|i| 0.05 * i as f64).collect();
    let current = Vector::zeros(fes.ndofs(1));
    let solution = solve_maxwell_leapfrog(fes, Medium::vacuum(), &times, initial, &current);

    // The magnetic charge starts at zero and must stay at zero throughout.
    for state in &solution {
      let charge = state.magnetic_charge(&ops).unwrap();
      assert_relative_eq!(charge.coeffs().norm(), 0.0, epsilon = 1e-11);
    }
  }

  /// Source-free electromagnetic energy is conserved by the leapfrog scheme.
  /// The staggered [`leapfrog_energy`] invariant is conserved to roundoff --
  /// the symplectic integrator has no energy drift at all, only the sloshing
  /// between the electric and magnetic reservoirs.
  #[test]
  fn energy_is_conserved() {
    let (topology, coords) = CartesianMeshInfo::new_unit(3, 2).compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);
    let fes = WhitneyComplex::new(&topology, &metric);
    let ops = MaxwellOperators::new(&fes, Medium::vacuum());

    let e0 = Cochain::new(1, seed_electric_field(fes.ndofs(1)));
    let b0 = Cochain::new(2, Vector::zeros(fes.ndofs(2)));
    let initial = MaxwellState::new(e0, b0);

    // A step size well within the CFL limit of this mesh.
    let dt = 0.02;
    let times: Vec<f64> = (0..=200).map(|i| dt * i as f64).collect();
    let current = Vector::zeros(fes.ndofs(1));
    let solution = solve_maxwell_leapfrog(fes, Medium::vacuum(), &times, initial, &current);

    let energy0 = leapfrog_energy(&solution[0], &solution[1], &ops);
    assert!(energy0 > 0.0);
    for pair in solution.windows(2) {
      let energy = leapfrog_energy(&pair[0], &pair[1], &ops);
      let drift = (energy - energy0).abs() / energy0;
      assert!(drift < 1e-9, "energy drifted by {drift}");
    }
  }
}
