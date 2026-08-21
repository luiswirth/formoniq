//! Structure-preserving time integration for the linear, constant-coefficient
//! semi-discrete systems that FEEC spatial assembly produces:
//!
//! $ M dot(y) = A y + f(t), $
//!
//! with $M$ the (constrained) mass matrix, $A$ a constant sparse operator, and
//! $f$ a possibly time-dependent forcing. This is the shape shared by
//! [`crate::problems::wave`] (position-velocity block system), the skew
//! Hodge–Dirac system of [`crate::problems::dirac`], and
//! [`crate::problems::heat`] after its affine boundary lifting.
//!
//! Two [`Tableau`] families serve the two structural regimes: Gauss-Legendre
//! collocation is symplectic and conserves every quadratic invariant of a
//! linear system exactly, the correct choice for the Hamiltonian systems
//! (wave, Hodge–Dirac Maxwell), where energy conservation, not just bounded
//! drift, is available. Radau IIA is L-stable and algebraically stable, the
//! correct choice for the dissipative heat equation, where the structure to
//! preserve is monotone decay under arbitrarily stiff eigenmodes, not
//! symplecticity.
//!
//! [`LinearIrk`] solves the coupled stage system directly (no Newton
//! iteration: $f$ is linear, so the stage equations are linear and one linear
//! solve is exact), assembled as a genuinely sparse block Kronecker system and
//! factored once for repeated stepping at a fixed $Delta t$.

use iterative::ApproxInverse;

use crate::linalg::{DirectInverse, faer::FaerLu, quadratic_form_sparse};
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

  /// Gauss-Legendre collocation, order $2s$ for any $s$: symplectic, and,
  /// the stronger fact that matters for a linear Hamiltonian system,
  /// exactly conserving every quadratic invariant, not merely a bounded
  /// shadow Hamiltonian.
  ///
  /// The nodes are the Gauss-Legendre quadrature points on $[0, 1]$ (roots of
  /// the shifted Legendre polynomial $P_s$); $A$ and $b$ then follow from the
  /// collocation conditions. $s = 1$ is the implicit midpoint rule.
  pub fn gauss_legendre(s: usize) -> Self {
    collocation_tableau(gauss_legendre_nodes(s))
  }

  /// Radau IIA collocation, order $2s - 1$ for any $s$: L-stable and
  /// algebraically stable, the dissipative counterpart to
  /// [`Self::gauss_legendre`].
  ///
  /// The nodes are the right Radau quadrature points on $[0, 1]$ (the last
  /// pinned at $c_s = 1$, which is what makes the method stiffly accurate);
  /// $A$ and $b$ follow from the collocation conditions. $s = 1$ is implicit
  /// (backward) Euler.
  pub fn radau_iia(s: usize) -> Self {
    collocation_tableau(radau_iia_nodes(s))
  }
}

/// The Butcher tableau of the collocation method on nodes $c in [0, 1]^s$: the
/// unique $(A, b)$ satisfying the simplified order conditions
///
/// $ sum_j a_(i j) c_j^(k-1) = c_i^k / k, quad
///   sum_i b_i c_i^(k-1) = 1 / k, quad k = 1, ..., s, $
///
/// i.e. $a_(i j) = integral_0^(c_i) ell_j$ and $b_i = integral_0^1 ell_i$ for
/// the Lagrange basis $ell_j$ on the nodes. Both are solves against the shared
/// node Vandermonde $V_(k j) = c_j^(k-1)$, so the whole tableau is fixed by the
/// nodes alone, the one construction of which Gauss-Legendre and Radau IIA
/// are the two node choices. The row-sum consistency $c_i = sum_j a_(i j)$ is
/// the $k = 1$ condition, so it holds by construction.
fn collocation_tableau(c: Vector) -> Tableau {
  let s = c.len();

  // Shared Vandermonde V_{kj} = c_j^k (row k is the order-k moment), factored
  // once for the b solve and all s rows of A.
  let mut vander = Matrix::zeros(s, s);
  for k in 0..s {
    for j in 0..s {
      vander[(k, j)] = c[j].powi(k as i32);
    }
  }
  let lu = vander.lu();

  let b_rhs = Vector::from_fn(s, |k, _| 1.0 / f64::from(k as u32 + 1));
  let b = lu
    .solve(&b_rhs)
    .expect("node Vandermonde is nonsingular for distinct collocation nodes");

  let mut a = Matrix::zeros(s, s);
  for i in 0..s {
    let a_rhs = Vector::from_fn(s, |k, _| c[i].powi(k as i32 + 1) / f64::from(k as u32 + 1));
    let a_row = lu
      .solve(&a_rhs)
      .expect("node Vandermonde is nonsingular for distinct collocation nodes");
    a.row_mut(i).copy_from(&a_row.transpose());
  }

  Tableau::new(a, b, c)
}

/// Gauss-Legendre quadrature nodes on $[0, 1]$: the $s$ eigenvalues of the
/// Legendre Golub-Welsch matrix (symmetric tridiagonal, zero diagonal,
/// off-diagonal $beta_j = j slash sqrt(4 j^2 - 1)$), mapped from $[-1, 1]$.
fn gauss_legendre_nodes(s: usize) -> Vector {
  map_unit(sorted_eigenvalues(legendre_jacobi(s)))
}

/// Right Radau quadrature nodes on $[0, 1]$: the Legendre Golub-Welsch matrix
/// with Golub's endpoint modification of the final diagonal entry, which pins
/// one eigenvalue at $x = 1$ (i.e. $c_s = 1$) while leaving the rest the Radau
/// points. Mapped from $[-1, 1]$.
fn radau_iia_nodes(s: usize) -> Vector {
  let mut jacobi = legendre_jacobi(s);
  if s >= 1 {
    // Golub (1973): to force the node a = 1, solve (T_{s-1} - a I) δ = β²_{s-1}
    // e_{s-1} on the leading block and set the last diagonal to a + δ_{s-1}.
    let a = 1.0;
    let last = s - 1;
    if s >= 2 {
      let beta = jacobi[(last, last - 1)];
      let block: Matrix =
        jacobi.view((0, 0), (last, last)).into_owned() - a * Matrix::identity(last, last);
      let mut e = Vector::zeros(last);
      e[last - 1] = beta * beta;
      let delta = block
        .lu()
        .solve(&e)
        .expect("Radau endpoint block is nonsingular (a is not an interior node)");
      jacobi[(last, last)] = a + delta[last - 1];
    } else {
      jacobi[(last, last)] = a;
    }
  }
  map_unit(sorted_eigenvalues(jacobi))
}

/// The Legendre Golub-Welsch (Jacobi) matrix of order $s$: symmetric
/// tridiagonal, zero diagonal, off-diagonal $beta_j = j slash sqrt(4 j^2 - 1)$.
/// Its eigenvalues are the Gauss-Legendre nodes on $[-1, 1]$.
fn legendre_jacobi(s: usize) -> Matrix {
  let mut t = Matrix::zeros(s, s);
  for j in 1..s {
    let beta = j as f64 / (4.0 * (j * j) as f64 - 1.0).sqrt();
    t[(j - 1, j)] = beta;
    t[(j, j - 1)] = beta;
  }
  t
}

/// Eigenvalues of a symmetric matrix, ascending.
fn sorted_eigenvalues(m: Matrix) -> Vector {
  let mut vals = m.symmetric_eigenvalues();
  vals.as_mut_slice().sort_by(f64::total_cmp);
  vals
}

/// Map quadrature nodes from $[-1, 1]$ to $[0, 1]$.
fn map_unit(x: Vector) -> Vector {
  x.map(|xi| 0.5 * (xi + 1.0))
}

/// Implicit Runge-Kutta time-stepper for the linear constant-coefficient
/// system $M dot(y) = A y + f(t)$.
///
/// Because $f$ is linear, the coupled stage system is linear and exact in one
/// solve, no Newton iteration. The stage system
///
/// $ (I_s ⊗ M - Delta t med (A_"tab" ⊗ A)) k = bb(1)_s ⊗ (A y_0) + f_"stage", $
///
/// is assembled directly as a sparse block matrix (never densified: each
/// block is a scaled copy of $M$'s or $A$'s own sparsity) and factored once,
/// since $M$, $A$ and $Delta t$ are all fixed across the integration, every
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
  /// $t_0 + c_i thin Delta t$.
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
/// $I_s ⊗ M - Delta t med (A_"tab" ⊗ A)$, assembled directly from the triplets of
/// $M$ and $A$, an $s times s$ grid of blocks, each block a scaled copy of
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

/// Explicit symplectic splitting integrator (Störmer–Verlet / velocity Verlet)
/// for the skew system $M dot(u) = A u$ with $M$ SPD and $A$ skew: the
/// dimension- and grade-general form of the Yee (FDTD) leapfrog.
///
/// The DOFs are 2-colored (`color[i]` the color of DOF $i$) so that $A$ couples
/// only across colors and $M$ only within them, a partition into which
/// the system splits as
///
/// $ M_0 dot(q) = A_(0 1) p, quad M_1 dot(p) = A_(1 0) q, quad A_(1 0) = -A_(0 1)^T, $
///
/// a linear Hamiltonian system with the two color blocks as canonical
/// position and momentum. On the Hodge–Dirac complex the coloring is grade
/// parity: $dif$ and $delta$ shift grade by one, so even and odd grades never
/// couple among themselves and $A$ is exactly this block-antidiagonal form,
/// leapfrog is then the discrete-time completion of the spatial staggering
/// (E on edges, B on faces) that FEEC already does.
///
/// One [`Self::step`] is a half-kick / drift / half-kick, co-located in time and
/// cheap: no solve ever involves $A$, only the two color-block masses $M_0, M_1$,
/// each Cholesky-factored once. Symplectic, so there is no energy drift; the
/// staggered invariant [`Self::conserved_energy`] is preserved to roundoff, and
/// is positive definite (a genuine norm, so the scheme is stable) precisely under
/// the CFL condition. Unlike [`LinearIrk`] this is only conditionally stable,
/// the price of being explicit.
pub struct Leapfrog {
  /// Global DOF indices of color 0 (the drifted "position" block $q$).
  idx0: Vec<usize>,
  /// Global DOF indices of color 1 (the half-kicked "momentum" block $p$).
  idx1: Vec<usize>,
  mass0: CsrMatrix,
  mass1: CsrMatrix,
  chol0: DirectInverse,
  chol1: DirectInverse,
  /// $A_(0 1)$: the color-0 $<-$ color-1 coupling, shape $n_0 times n_1$.
  a01: CsrMatrix,
  /// $A_(1 0) = -A_(0 1)^T$: the color-1 $<-$ color-0 coupling.
  a10: CsrMatrix,
  dt: f64,
  ndofs: usize,
}

impl Leapfrog {
  pub fn new(mass: &CsrMatrix, op: &CsrMatrix, color: &[bool], dt: f64) -> Self {
    let ndofs = mass.nrows();
    assert_eq!(mass.ncols(), ndofs);
    assert_eq!(op.nrows(), ndofs);
    assert_eq!(op.ncols(), ndofs);
    assert_eq!(color.len(), ndofs);

    let idx0: Vec<usize> = (0..ndofs).filter(|&i| !color[i]).collect();
    let idx1: Vec<usize> = (0..ndofs).filter(|&i| color[i]).collect();
    let (n0, n1) = (idx0.len(), idx1.len());

    // Global -> local index within a DOF's own color block.
    let mut local = vec![0usize; ndofs];
    for (l, &g) in idx0.iter().enumerate() {
      local[g] = l;
    }
    for (l, &g) in idx1.iter().enumerate() {
      local[g] = l;
    }

    // The coloring is only valid if $M$ does not couple the two colors and $A$
    // does not couple within a color; otherwise the split is not the required
    // block-antidiagonal form and the method is neither explicit nor correct.
    for (r, c, &v) in mass.triplet_iter() {
      assert!(
        v == 0.0 || color[r] == color[c],
        "mass couples the two colors"
      );
    }
    for (r, c, &v) in op.triplet_iter() {
      assert!(
        v == 0.0 || color[r] != color[c],
        "operator couples within a color"
      );
    }

    let block = |src: &CsrMatrix, want_r: bool, want_c: bool, nr: usize, nc: usize| {
      let mut coo = CooMatrix::new(nr, nc);
      for (r, c, &v) in src.triplet_iter() {
        if color[r] == want_r && color[c] == want_c {
          coo.push(local[r], local[c], v);
        }
      }
      CsrMatrix::from(&coo)
    };

    let mass0 = block(mass, false, false, n0, n0);
    let mass1 = block(mass, true, true, n1, n1);
    let a01 = block(op, false, true, n0, n1);
    let a10 = block(op, true, false, n1, n0);

    Self {
      idx0,
      idx1,
      chol0: DirectInverse::new(mass0.clone()),
      chol1: DirectInverse::new(mass1.clone()),
      mass0,
      mass1,
      a01,
      a10,
      dt,
      ndofs,
    }
  }

  fn gather(&self, y: &Vector, idx: &[usize]) -> Vector {
    Vector::from_iterator(idx.len(), idx.iter().map(|&g| y[g]))
  }
  fn scatter(&self, y: &mut Vector, idx: &[usize], v: &Vector) {
    for (l, &g) in idx.iter().enumerate() {
      y[g] = v[l];
    }
  }

  /// Advance `y0` by one step of the fixed `dt`, co-located in time: a half-kick
  /// of $p$, a full drift of $q$, and a closing half-kick of $p$.
  pub fn step(&self, y0: &Vector) -> Vector {
    let mut q = self.gather(y0, &self.idx0);
    let mut p = self.gather(y0, &self.idx1);

    p += 0.5 * self.dt * self.chol1.apply(&(&self.a10 * &q));
    q += self.dt * self.chol0.apply(&(&self.a01 * &p));
    p += 0.5 * self.dt * self.chol1.apply(&(&self.a10 * &q));

    let mut y1 = Vector::zeros(self.ndofs);
    self.scatter(&mut y1, &self.idx0, &q);
    self.scatter(&mut y1, &self.idx1, &p);
    y1
  }

  /// The exactly conserved staggered invariant
  ///
  /// $ E = 1/2 q^T M_0 q + 1/2 p^T M_1 p
  ///     - dif t^2/8 (A_(1 0) q)^T M_1^(-1) (A_(1 0) q), $
  ///
  /// the co-located energy $1/2 u^T M u$ minus the leapfrog defect. Preserved to
  /// roundoff by [`Self::step`] (unlike the co-located energy, which oscillates
  /// at $O(dif t^2)$), and positive definite, a norm, whence stability,
  /// precisely under the CFL condition.
  pub fn conserved_energy(&self, y: &Vector) -> f64 {
    let q = self.gather(y, &self.idx0);
    let p = self.gather(y, &self.idx1);
    let a10q = &self.a10 * &q;
    let defect = a10q.dot(&self.chol1.apply(&a10q));
    0.5 * quadratic_form_sparse(&self.mass0, &q) + 0.5 * quadratic_form_sparse(&self.mass1, &p)
      - self.dt * self.dt / 8.0 * defect
  }
}
