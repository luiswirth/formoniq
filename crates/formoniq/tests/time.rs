//! Structure-preserving laws for [`formoniq::time`]: Gauss-Legendre conserves
//! a quadratic invariant exactly, Radau IIA is stiffly accurate and so solves
//! the index-1 DAE a singular mass produces, and the explicit leapfrog is
//! symplectic.

use approx::assert_relative_eq;
use formoniq::time::{Leapfrog, LinearIrk, Tableau};
use simplicial::linalg::{CooMatrix, CsrMatrix, Vector};

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
/// quadratic invariant $H = 1/2 (v^2 + omega^2 x^2)$, to roundoff, not
/// merely bounded, across many periods and stage counts.
#[test]
fn gauss_legendre_conserves_energy_exactly() {
  let omega = 1.7;
  let op = oscillator(omega);
  let mass = identity(2);

  for s in 1..=4 {
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

/// A singular mass matrix turns $M dot(y) = A y$ into an index-1
/// differential-algebraic system: the shape the mixed Hodge-Laplace
/// evolution problems produce, where the auxiliary $sigma = delta u$ carries
/// no time derivative. The 1-dof model is $sigma = u$ (algebraic),
/// $dot(u) = -lambda sigma$, i.e. $M = mat(0,0;0,1)$,
/// $A = mat(-1,1;-lambda,0)$, whose $u$-component is the exact decay
/// $u(t) = u_0 e^(-lambda t)$. Radau IIA is stiffly accurate, so it solves
/// the algebraic constraint at every stage and reproduces the decay even
/// though $M$ is not invertible, the fact the heat and wave solvers rely
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

/// The explicit leapfrog is symplectic on the skew system $M dot(u) = A u$ it
/// targets: its staggered invariant is preserved to roundoff (no drift) across
/// many periods within the CFL limit. A 2-dof skew system with distinct block
/// masses, 2-colored into position (dof 0) and momentum (dof 1), the minimal
/// model of the grade-parity split.
#[test]
fn leapfrog_conserves_staggered_energy_exactly() {
  // 2 q' = p, 3 p' = -q: M = diag(2, 3), A = [[0, 1], [-1, 0]] (skew).
  let mut m = CooMatrix::new(2, 2);
  m.push(0, 0, 2.0);
  m.push(1, 1, 3.0);
  let mass = CsrMatrix::from(&m);
  let mut a = CooMatrix::new(2, 2);
  a.push(0, 1, 1.0);
  a.push(1, 0, -1.0);
  let op = CsrMatrix::from(&a);
  let color = [false, true];

  let dt = 0.2;
  let lf = Leapfrog::new(&mass, &op, &color, dt);

  let mut y = Vector::from_row_slice(&[1.0, 0.5]);
  let e0 = lf.conserved_energy(&y);
  assert!(e0 > 0.0);
  for _ in 0..1000 {
    y = lf.step(&y);
    assert_relative_eq!(lf.conserved_energy(&y), e0, epsilon = 1e-10 * e0.max(1.0));
  }
}
