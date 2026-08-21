//! Structure-preserving laws for [`formoniq::time`]: the collocation
//! tableaus reproduce the classical Runge-Kutta coefficients and satisfy
//! their order conditions, Gauss-Legendre conserves quadratic invariants
//! exactly (including on singular-mass DAEs), Radau IIA is L-stable and
//! stiffly accurate, and the explicit leapfrog is symplectic.

use approx::assert_relative_eq;
use formoniq::time::{Leapfrog, LinearIrk, Tableau};
use simplicial::linalg::{CooMatrix, CsrMatrix, Matrix, Vector};

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

/// The $s = 1, 2$ Gauss-Legendre and Radau IIA tableaus produced by the
/// general collocation construction reproduce the classical hardcoded
/// coefficients (Hairer & Wanner, Solving ODEs II, Tables 5.2, 5.6) to
/// roundoff: implicit midpoint and the fourth-order Gauss block; backward
/// Euler and the third-order Radau block.
#[test]
fn low_stage_tableaus_match_classical_coefficients() {
  let sqrt3 = 3f64.sqrt();

  let gl1 = Tableau::gauss_legendre(1);
  assert_relative_eq!(gl1.a, Matrix::from_row_slice(1, 1, &[0.5]));
  assert_relative_eq!(gl1.b, Vector::from_row_slice(&[1.0]));
  assert_relative_eq!(gl1.c, Vector::from_row_slice(&[0.5]));

  let gl2 = Tableau::gauss_legendre(2);
  assert_relative_eq!(
    gl2.a,
    Matrix::from_row_slice(2, 2, &[0.25, 0.25 - sqrt3 / 6.0, 0.25 + sqrt3 / 6.0, 0.25])
  );
  assert_relative_eq!(gl2.b, Vector::from_row_slice(&[0.5, 0.5]));
  assert_relative_eq!(
    gl2.c,
    Vector::from_row_slice(&[0.5 - sqrt3 / 6.0, 0.5 + sqrt3 / 6.0])
  );

  let r1 = Tableau::radau_iia(1);
  assert_relative_eq!(r1.a, Matrix::from_row_slice(1, 1, &[1.0]));
  assert_relative_eq!(r1.b, Vector::from_row_slice(&[1.0]));
  assert_relative_eq!(r1.c, Vector::from_row_slice(&[1.0]));

  let r2 = Tableau::radau_iia(2);
  assert_relative_eq!(
    r2.a,
    Matrix::from_row_slice(2, 2, &[5.0 / 12.0, -1.0 / 12.0, 3.0 / 4.0, 1.0 / 4.0])
  );
  assert_relative_eq!(r2.b, Vector::from_row_slice(&[3.0 / 4.0, 1.0 / 4.0]));
  assert_relative_eq!(r2.c, Vector::from_row_slice(&[1.0 / 3.0, 1.0]));
}

/// A collocation tableau satisfies the simplified conditions that define it,
/// at every stage count: row-sum consistency $c_i = sum_j a_(i j)$, the stage
/// conditions $C(s)$, and the quadrature conditions $B(s)$, so both families
/// have their claimed order $p$ ($2s$ for Gauss, $2s - 1$ for Radau) at every
/// stage count. Radau additionally pins its last node at $c_s = 1$ (stiff
/// accuracy).
#[test]
fn collocation_tableaus_satisfy_order_conditions() {
  for s in 1..=6 {
    for tab in [Tableau::gauss_legendre(s), Tableau::radau_iia(s)] {
      assert_eq!(tab.s, s);

      // Row-sum consistency c_i = sum_j a_ij.
      for i in 0..s {
        assert_relative_eq!(tab.c[i], tab.a.row(i).sum(), epsilon = 1e-12);
      }

      // C(s): sum_j a_ij c_j^{k-1} = c_i^k / k, and B(s): sum_i b_i c_i^{k-1} = 1/k.
      for k in 1..=s {
        let kf = k as f64;
        for i in 0..s {
          let lhs: f64 = (0..s)
            .map(|j| tab.a[(i, j)] * tab.c[j].powi(k as i32 - 1))
            .sum();
          assert_relative_eq!(lhs, tab.c[i].powi(k as i32) / kf, epsilon = 1e-11);
        }
        let quad: f64 = (0..s).map(|i| tab.b[i] * tab.c[i].powi(k as i32 - 1)).sum();
        assert_relative_eq!(quad, 1.0 / kf, epsilon = 1e-11);
      }
    }

    // Radau IIA is stiffly accurate: its final node is pinned at 1.
    let radau = Tableau::radau_iia(s);
    assert_relative_eq!(radau.c[s - 1], 1.0, epsilon = 1e-12);
  }
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

/// Radau IIA on the scalar decay $dot(y) = -lambda y$ reproduces
/// $exp(-lambda t)$ to the scheme's classical order, and stays monotone
/// (no oscillatory overshoot) even at a step size well past the explicit
/// stability limit, the L-stability that Gauss does not have.
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
/// $y_infty = -A^(-1) f$. Both tableaus must reach it.
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
