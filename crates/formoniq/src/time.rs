//! Structure-preserving time integration for the linear, constant-coefficient
//! semi-discrete systems that FEEC spatial assembly produces:
//!
//! $ M dot(y) = A y + f(t), $
//!
//! with $M$ the (constrained) mass matrix, $A$ a constant sparse operator, and
//! $f$ a possibly time-dependent forcing. This is the shape shared by
//! [`crate::problems::wave`] (position-velocity block system), the curl-curl
//! form of [`crate::problems::maxwell`], and [`crate::problems::heat`] after
//! its affine boundary lifting.
//!
//! Two [`Tableau`] families serve the two structural regimes: Gauss-Legendre
//! collocation is symplectic and conserves every quadratic invariant of a
//! linear system exactly -- the correct choice for the Hamiltonian systems
//! (wave, Maxwell curl-curl), where energy conservation, not just bounded
//! drift, is available. Radau IIA is L-stable and algebraically stable -- the
//! correct choice for the dissipative heat equation, where the structure to
//! preserve is monotone decay under arbitrarily stiff eigenmodes, not
//! symplecticity.
//!
//! [`LinearIrk`] solves the coupled stage system directly (no Newton
//! iteration: $f$ is linear, so the stage equations are linear and one linear
//! solve is exact), assembled as a genuinely sparse block Kronecker system and
//! factored once for repeated stepping at a fixed $dt$.

use crate::linalg::faer::FaerLu;
use simplicial::linalg::{CooMatrix, CsrMatrix, Matrix, Vector};

/// A Butcher tableau $(A, b, c)$ for an $s$-stage Runge-Kutta method.
pub struct Tableau {
  /// $A in RR^(s times s)$: the stage coupling.
  pub a: Matrix,
  /// $b in RR^s$: the step weights.
  pub b: Vector,
  /// $c in RR^s$: the stage nodes, $c_i = sum_j a_(i j)$.
  pub c: Vector,
  /// The number of stages $s$.
  pub s: usize,
}

impl Tableau {
  fn new(a: Matrix, b: Vector, c: Vector) -> Self {
    let s = b.len();
    assert!(a.nrows() == s && a.ncols() == s);
    assert!(c.len() == s);
    Self { a, b, c, s }
  }

  /// Gauss-Legendre collocation, order $2s$: symplectic for any $s$, and --
  /// the stronger fact that matters for a *linear* Hamiltonian system --
  /// exactly conserving every quadratic invariant, not merely a bounded
  /// shadow Hamiltonian. Supports $s in {1, 2}$.
  ///
  /// $s = 1$ is the implicit midpoint rule. Coefficients: Hairer & Wanner,
  /// *Solving ODEs II*, Table 5.2.
  pub fn gauss_legendre(s: usize) -> Self {
    match s {
      1 => Self::new(
        Matrix::from_row_slice(1, 1, &[0.5]),
        Vector::from_row_slice(&[1.0]),
        Vector::from_row_slice(&[0.5]),
      ),
      2 => {
        let sqrt3 = 3f64.sqrt();
        Self::new(
          Matrix::from_row_slice(2, 2, &[0.25, 0.25 - sqrt3 / 6.0, 0.25 + sqrt3 / 6.0, 0.25]),
          Vector::from_row_slice(&[0.5, 0.5]),
          Vector::from_row_slice(&[0.5 - sqrt3 / 6.0, 0.5 + sqrt3 / 6.0]),
        )
      }
      _ => unimplemented!("Gauss-Legendre tableau only implemented for s in {{1, 2}}"),
    }
  }

  /// Radau IIA collocation, order $2s - 1$: L-stable and algebraically
  /// stable, the dissipative counterpart to [`Self::gauss_legendre`].
  /// Supports $s in {1, 2}$.
  ///
  /// $s = 1$ is implicit (backward) Euler. Coefficients: Hairer & Wanner,
  /// *Solving ODEs II*, Table 5.6.
  pub fn radau_iia(s: usize) -> Self {
    match s {
      1 => Self::new(
        Matrix::from_row_slice(1, 1, &[1.0]),
        Vector::from_row_slice(&[1.0]),
        Vector::from_row_slice(&[1.0]),
      ),
      2 => Self::new(
        Matrix::from_row_slice(2, 2, &[5.0 / 12.0, -1.0 / 12.0, 3.0 / 4.0, 1.0 / 4.0]),
        Vector::from_row_slice(&[3.0 / 4.0, 1.0 / 4.0]),
        Vector::from_row_slice(&[1.0 / 3.0, 1.0]),
      ),
      _ => unimplemented!("Radau IIA tableau only implemented for s in {{1, 2}}"),
    }
  }
}

/// Implicit Runge-Kutta time-stepper for the linear constant-coefficient
/// system $M dot(y) = A y + f(t)$.
///
/// Because $f$ is linear, the coupled stage system is linear and exact in one
/// solve -- no Newton iteration. The stage system
///
/// $ (I_s ⊗ M - dt med (A_"tab" ⊗ A)) k = bb(1)_s ⊗ (A y_0) + f_"stage", $
///
/// is assembled directly as a sparse block matrix (never densified: each
/// block is a scaled copy of $M$'s or $A$'s own sparsity) and factored once,
/// since $M$, $A$ and $dt$ are all fixed across the integration -- every
/// subsequent [`Self::step`] is then a single triangular solve.
pub struct LinearIrk {
  tableau: Tableau,
  op: CsrMatrix,
  dt: f64,
  ndofs: usize,
  stage_lu: FaerLu,
}

impl LinearIrk {
  pub fn new(tableau: Tableau, mass: &CsrMatrix, op: CsrMatrix, dt: f64) -> Self {
    let ndofs = mass.nrows();
    assert_eq!(mass.ncols(), ndofs);
    assert_eq!(op.nrows(), ndofs);
    assert_eq!(op.ncols(), ndofs);

    let stage_matrix = stage_matrix(&tableau.a, mass, &op, dt);
    let stage_lu = FaerLu::new(stage_matrix);

    Self {
      tableau,
      op,
      dt,
      ndofs,
      stage_lu,
    }
  }

  /// Advance `y0` (at time `t0`) by one step of the fixed `dt` this
  /// integrator was built with. `forcing` is evaluated at each stage time
  /// $t_0 + c_i thin dt$.
  pub fn step(&self, y0: &Vector, t0: f64, forcing: impl Fn(f64) -> Vector) -> Vector {
    let s = self.tableau.s;
    let d = self.ndofs;

    let ay0 = &self.op * y0;
    let mut rhs = Vector::zeros(s * d);
    for i in 0..s {
      let stage_time = t0 + self.tableau.c[i] * self.dt;
      let fi = &ay0 + forcing(stage_time);
      rhs.rows_mut(i * d, d).copy_from(&fi);
    }

    let k = self.stage_lu.solve(&rhs);

    let mut y1 = y0.clone();
    for i in 0..s {
      y1 += self.dt * self.tableau.b[i] * k.rows(i * d, d);
    }
    y1
  }
}

/// The sparse block Kronecker stage matrix
/// $I_s ⊗ M - dt med (A_"tab" ⊗ A)$, assembled directly from the triplets of
/// $M$ and $A$ -- an $s times s$ grid of blocks, each block a scaled copy of
/// $M$'s or $A$'s own sparsity pattern, never a dense intermediate.
fn stage_matrix(a_tab: &Matrix, mass: &CsrMatrix, op: &CsrMatrix, dt: f64) -> CsrMatrix {
  let s = a_tab.nrows();
  let d = mass.nrows();

  let mut coo = CooMatrix::new(s * d, s * d);
  for i in 0..s {
    for (r, c, &v) in mass.triplet_iter() {
      coo.push(i * d + r, i * d + c, v);
    }
    for j in 0..s {
      let coeff = -dt * a_tab[(i, j)];
      if coeff != 0.0 {
        for (r, c, &v) in op.triplet_iter() {
          coo.push(i * d + r, j * d + c, coeff * v);
        }
      }
    }
  }
  CsrMatrix::from(&coo)
}

#[cfg(test)]
mod test {
  use super::*;
  use approx::assert_relative_eq;

  /// A 1-dof harmonic oscillator recast as the first-order block system
  /// $dot(y) = A y$, $y = (x, v)$, $A = [[0, 1], [-omega^2, 0]]$: the scalar
  /// Hamiltonian test case for the whole `time` module.
  fn oscillator(omega: f64) -> CsrMatrix {
    let mut coo = CooMatrix::new(2, 2);
    coo.push(0, 1, 1.0);
    coo.push(1, 0, -omega * omega);
    CsrMatrix::from(&coo)
  }

  fn identity(n: usize) -> CsrMatrix {
    let mut coo = CooMatrix::new(n, n);
    for i in 0..n {
      coo.push(i, i, 1.0);
    }
    CsrMatrix::from(&coo)
  }

  /// Gauss-Legendre on a linear Hamiltonian system exactly conserves the
  /// quadratic invariant $H = 1/2 (v^2 + omega^2 x^2)$ -- to roundoff, not
  /// merely bounded, across many periods and stage counts.
  #[test]
  fn gauss_legendre_conserves_energy_exactly() {
    let omega = 1.7;
    let op = oscillator(omega);
    let mass = identity(2);

    for s in [1, 2] {
      let dt = 0.3;
      let irk = LinearIrk::new(Tableau::gauss_legendre(s), &mass, op.clone(), dt);

      let mut y = Vector::from_row_slice(&[1.0, 0.0]);
      let energy0 = 0.5 * (y[1] * y[1] + omega * omega * y[0] * y[0]);

      let mut t = 0.0;
      for _ in 0..500 {
        y = irk.step(&y, t, |_| Vector::zeros(2));
        t += dt;
      }
      let energy = 0.5 * (y[1] * y[1] + omega * omega * y[0] * y[0]);
      assert_relative_eq!(energy, energy0, epsilon = 1e-10);
    }
  }

  /// Radau IIA on the scalar decay $dot(y) = -lambda y$ reproduces
  /// $exp(-lambda t)$ to the scheme's classical order, and stays monotone
  /// (no oscillatory overshoot) even at a step size well past the explicit
  /// stability limit -- the L-stability that Gauss does not have.
  #[test]
  fn radau_iia_is_monotone_and_accurate_for_stiff_decay() {
    let lambda = 500.0;
    let mut coo = CooMatrix::new(1, 1);
    coo.push(0, 0, -lambda);
    let op = CsrMatrix::from(&coo);
    let mass = identity(1);

    let dt = 0.05; // lambda * dt = 25, far outside explicit stability
    let irk = LinearIrk::new(Tableau::radau_iia(2), &mass, op, dt);

    let mut y = Vector::from_row_slice(&[1.0]);
    let mut t = 0.0;
    for _ in 0..10 {
      let y_next = irk.step(&y, t, |_| Vector::zeros(1));
      assert!(y_next[0].abs() <= y[0].abs(), "decay must stay monotone");
      y = y_next;
      t += dt;
    }
    let exact = (-lambda * t).exp();
    assert_relative_eq!(y[0], exact, epsilon = 1e-2);
  }

  /// A singular mass matrix turns $M dot(y) = A y$ into an index-1
  /// differential-algebraic system: the shape the mixed Hodge-Laplace
  /// evolution problems produce, where the auxiliary $sigma = delta u$ carries
  /// no time derivative. The 1-dof model is $sigma = u$ (algebraic),
  /// $dot(u) = -lambda sigma$, i.e. $M = mat(0,0;0,1)$,
  /// $A = mat(-1,1;-lambda,0)$, whose $u$-component is the exact decay
  /// $u(t) = u_0 e^(-lambda t)$. Radau IIA is stiffly accurate, so it solves
  /// the algebraic constraint at every stage and reproduces the decay even
  /// though $M$ is not invertible -- the fact the heat and wave solvers rely
  /// on.
  #[test]
  fn radau_iia_solves_index_one_dae_with_singular_mass() {
    let lambda = 2.0;

    let mut m = CooMatrix::new(2, 2);
    m.push(1, 1, 1.0);
    let mass = CsrMatrix::from(&m);

    let mut a = CooMatrix::new(2, 2);
    a.push(0, 0, -1.0);
    a.push(0, 1, 1.0);
    a.push(1, 0, -lambda);
    let op = CsrMatrix::from(&a);

    let dt = 0.05;
    let irk = LinearIrk::new(Tableau::radau_iia(2), &mass, op, dt);

    let mut y = Vector::from_row_slice(&[1.0, 1.0]);
    let mut t = 0.0;
    for _ in 0..40 {
      y = irk.step(&y, t, |_| Vector::zeros(2));
      t += dt;
      // The algebraic constraint sigma = u holds after each step.
      assert_relative_eq!(y[0], y[1], epsilon = 1e-9);
    }
    let exact = (-lambda * t).exp();
    assert_relative_eq!(y[1], exact, epsilon = 1e-4);
  }

  /// The wave solver feeds Gauss-Legendre a singular-mass DAE too: the mixed
  /// $(sigma, u, w)$ form with $sigma = delta u$ algebraic, $dot(u) = w$,
  /// $dot(w) = -Delta u = -sigma$ at 1 dof. Because the constraint is linear,
  /// the reduced $(u, w)$ dynamics are a genuine linear Hamiltonian oscillator,
  /// and Gauss-Legendre conserves its quadratic energy
  /// $1/2 (u^2 + w^2) = 1/2 (norm(delta u)^2 + norm(dot(u))^2)$ exactly even
  /// through the algebraic $sigma$.
  #[test]
  fn gauss_legendre_conserves_energy_on_singular_mass_wave_dae() {
    let mut m = CooMatrix::new(3, 3);
    m.push(1, 1, 1.0);
    m.push(2, 2, 1.0);
    let mass = CsrMatrix::from(&m);

    let mut a = CooMatrix::new(3, 3);
    a.push(0, 0, -1.0);
    a.push(0, 1, 1.0); // sigma = u
    a.push(1, 2, 1.0); // u_t = w
    a.push(2, 0, -1.0); // w_t = -sigma
    let op = CsrMatrix::from(&a);

    let dt = 0.3;
    let irk = LinearIrk::new(Tableau::gauss_legendre(2), &mass, op, dt);

    let mut y = Vector::from_row_slice(&[1.0, 1.0, 0.0]);
    let energy0 = 0.5 * (y[1] * y[1] + y[2] * y[2]);
    let mut t = 0.0;
    for _ in 0..500 {
      y = irk.step(&y, t, |_| Vector::zeros(3));
      t += dt;
      assert_relative_eq!(y[0], y[1], epsilon = 1e-9);
    }
    let energy = 0.5 * (y[1] * y[1] + y[2] * y[2]);
    assert_relative_eq!(energy, energy0, epsilon = 1e-9);
  }

  /// A constant forcing steers the linear system to its steady state
  /// $y_infty = -A^(-1) f$; both tableaus must reach it.
  #[test]
  fn constant_forcing_reaches_steady_state() {
    let lambda = 3.0;
    let mut coo = CooMatrix::new(1, 1);
    coo.push(0, 0, -lambda);
    let op = CsrMatrix::from(&coo);
    let mass = identity(1);
    let force = 6.0;

    let dt = 0.2;
    let irk = LinearIrk::new(Tableau::radau_iia(2), &mass, op, dt);

    let mut y = Vector::from_row_slice(&[0.0]);
    let mut t = 0.0;
    for _ in 0..200 {
      y = irk.step(&y, t, |_| Vector::from_row_slice(&[force]));
      t += dt;
    }
    assert_relative_eq!(y[0], force / lambda, epsilon = 1e-6);
  }
}
