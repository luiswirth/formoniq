#![doc = include_str!("../README.md")]

extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

pub mod aux_space;
pub mod krylov;
pub mod multigrid;
mod operator;
mod precond;
pub mod stationary;

pub use aux_space::AuxiliarySpace;
pub use multigrid::{Level, VCycle};
pub use precond::{BlockDiagonal, Identity, Jacobi};
pub use stationary::Stationary;

/// A dense real vector, the currency of every apply.
pub type Vector = na::DVector<f64>;
/// A sparse real matrix in compressed-row storage: the assembled operator, and
/// the source a diagonal or triangular preconditioner reads its entries from.
pub type CsrMatrix = nas::CsrMatrix<f64>;

/// A real inner product space: the structure a Krylov method asks of its
/// vectors, and nothing more.
///
/// Conjugate gradients and MINRES scale a vector, add a scaled one to it, and
/// take inner products. They never index an entry, never slice, never name a
/// basis, and the only vector they can produce without being handed one is
/// zero. That is exactly this trait, so a method written against it runs
/// wherever its vectors live.
///
/// [`Vector`] is the archetypal instance. The reason to want another is a
/// vector that does not live in host memory, where copying the iterates back
/// and forth every step would cost more than the solve; pinning the space in
/// the operator makes pairing one with a preconditioner living elsewhere a
/// compile error rather than an implicit transfer.
///
/// The zero vector is taken from an existing one rather than from a dimension,
/// because a dimension does not determine a vector: a device vector needs an
/// allocator to exist, and the solution of $A x = b$ lives in the space $b$
/// does. That is also why the trait carries no dimension of its own.
pub trait InnerProductSpace: Clone {
  /// The zero vector of the space this one lives in: the additive identity,
  /// and the only element a Krylov method can name without being handed one.
  fn zeros_like(&self) -> Self;
  /// The inner product $angle.l x, y angle.r$.
  fn dot(&self, other: &Self) -> f64;
  /// $x <- alpha x$, the scalar action.
  fn scale(&mut self, alpha: f64);
  /// $y <- y + alpha x$, the addition, fused with the scaling of its argument.
  ///
  /// Fused rather than composed out of [`scale`](Self::scale) and
  /// [`add`](Self::add) because a Krylov method never asks for bare addition:
  /// every update it makes is of this shape, and building each one from a
  /// scaled copy would allocate a vector per step.
  ///
  /// `x` is a distinct vector from `self`.
  fn add_scaled(&mut self, alpha: f64, x: &Self);
  /// $y <- y + x$.
  fn add(&mut self, x: &Self) {
    self.add_scaled(1.0, x);
  }
  /// The induced norm $norm(x) = sqrt(angle.l x, x angle.r)$.
  fn norm(&self) -> f64 {
    self.dot(self).sqrt()
  }
}

impl InnerProductSpace for Vector {
  fn zeros_like(&self) -> Self {
    Vector::zeros(self.len())
  }
  fn dot(&self, other: &Self) -> f64 {
    na::DVector::dot(self, other)
  }
  fn scale(&mut self, alpha: f64) {
    *self *= alpha;
  }
  fn add_scaled(&mut self, alpha: f64, x: &Self) {
    self.axpy(alpha, x, 1.0);
  }
}

/// A linear operator $A: V -> V$, applied as $x |-> A x$ --- the only
/// thing a Krylov method asks of its system matrix.
///
/// Distinct from [`ApproxInverse`] by intent, not by shape: this is $A$, that is
/// $B approx A^(-1)$. Entry-needing preconditioners (a diagonal, a triangular
/// sweep) take the assembled [`CsrMatrix`] at construction instead, which keeps
/// this interface at the matrix-free minimum a matvec needs.
pub trait LinearOperator {
  /// The space the operator acts on.
  type Space: InnerProductSpace;
  /// The order $n$ of the square operator.
  fn dim(&self) -> usize;
  /// Apply the operator: $x |-> A x$.
  fn apply(&self, x: &Self::Space) -> Self::Space;
}

/// A cheap approximate inverse $B approx A^(-1)$, applied as $r |-> B r$.
///
/// One object, three roles, differing only in which consumer holds it: iterated
/// alone it is a *solver*; wrapped in a Krylov method it is a *preconditioner*;
/// sitting on a level of a multigrid hierarchy it is a *smoother*. The exact
/// inverse $B = A^(-1)$ (a factorization) is the perfect special case, and the
/// identity $B = I$ the trivial one.
pub trait ApproxInverse {
  /// The space the approximate inverse acts on, which a solver requires to be
  /// the operator's own.
  type Space: InnerProductSpace;
  /// The order $n$ of the square operator approximated.
  fn dim(&self) -> usize;
  /// Apply the approximate inverse: $r |-> B r$.
  fn apply(&self, r: &Self::Space) -> Self::Space;
}

/// Marker: [`ApproxInverse::apply`] is a fixed self-adjoint positive-definite
/// linear operator, $angle.l B r, s angle.r = angle.l r, B s angle.r$ and
/// $angle.l B r, r angle.r > 0$.
///
/// The precondition a symmetric Krylov method rests on. Conjugate gradients
/// takes its preconditioner only through this bound, so a non-symmetric
/// approximate inverse (a single forward Gauss-Seidel sweep, a V-cycle with
/// asymmetric smoothing) is rejected at compile time rather than converging
/// erratically at runtime. Self-adjointness is structural; positive-definiteness
/// is the constructor's promise, exactly as for the operator it approximates.
pub trait SelfAdjoint: ApproxInverse {}

/// When to stop iterating: a relative residual target and an iteration ceiling.
#[derive(Clone, Copy, Debug)]
pub struct StopCriterion {
  /// Stop once $norm(r_k) <= "rtol" dot norm(b)$.
  pub rtol: f64,
  /// Stop unconditionally after this many iterations.
  pub max_iters: usize,
}

impl StopCriterion {
  /// A relative-residual target with a generous iteration ceiling.
  pub fn rtol(rtol: f64) -> Self {
    Self {
      rtol,
      max_iters: 10_000,
    }
  }
}

/// The outcome of a solve: how far it got and whether it met the tolerance.
#[derive(Clone, Copy, Debug)]
pub struct Report {
  /// Iterations actually taken.
  pub iters: usize,
  /// The final relative residual $norm(r) / norm(b)$.
  pub residual: f64,
  /// Whether [`StopCriterion::rtol`] was met before the ceiling.
  pub converged: bool,
}

#[cfg(test)]
mod testutil {
  use crate::{CsrMatrix, Vector};
  use na::DMatrix;

  /// Sparse operator from a dense one, via triplets. Small-system test glue.
  pub fn csr(dense: &DMatrix<f64>) -> CsrMatrix {
    let (r, c) = dense.shape();
    let mut coo = nas::CooMatrix::new(r, c);
    for j in 0..c {
      for i in 0..r {
        let v = dense[(i, j)];
        if v != 0.0 {
          coo.push(i, j, v);
        }
      }
    }
    CsrMatrix::from(&coo)
  }

  /// A symmetric positive-definite operator with a prescribed spectrum,
  /// $A = Q "diag"(lambda) Q^T$ with $Q$ a deterministic orthogonal factor.
  /// Controlled conditioning: the finite-termination law degrades under an
  /// ill-conditioned random matrix, so the spectrum is pinned, not sampled.
  pub fn spd_from_spectrum(eigs: &[f64]) -> DMatrix<f64> {
    let n = eigs.len();
    let seed = DMatrix::from_fn(n, n, |i, j| ((i * 7 + j * 13) % 11) as f64 - 5.0);
    let q = seed.qr().q();
    let lambda = DMatrix::from_diagonal(&Vector::from_column_slice(eigs));
    &q * lambda * q.transpose()
  }

  /// The tridiagonal $"diag" I - "off" (L + L^T)$: SPD, and strictly diagonally
  /// dominant for $"diag" > 2 "off"$, so a Jacobi sweep is a contraction.
  pub fn tridiag(n: usize, diag: f64, off: f64) -> DMatrix<f64> {
    DMatrix::from_fn(n, n, |i, j| {
      if i == j {
        diag
      } else if i.abs_diff(j) == 1 {
        -off
      } else {
        0.0
      }
    })
  }

  /// A symmetric *indefinite* operator: same construction as
  /// [`spd_from_spectrum`] but with a prescribed mixed-sign spectrum, so it is
  /// the MINRES-shaped case CG cannot handle.
  pub fn symmetric_from_spectrum(eigs: &[f64]) -> DMatrix<f64> {
    spd_from_spectrum(eigs)
  }

  /// Direct dense solve, the reference an iterative method must reproduce.
  pub fn dense_solve(a: &DMatrix<f64>, b: &Vector) -> Vector {
    a.clone().lu().solve(b).expect("nonsingular")
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::testutil::{csr, dense_solve, spd_from_spectrum, tridiag};
  use na::DMatrix;

  #[test]
  fn csr_matvec_matches_dense() {
    let dense = spd_from_spectrum(&[1.0, 2.0, 3.0, 4.0]);
    let a = csr(&dense);
    let x = Vector::from_column_slice(&[1.0, -2.0, 0.5, 3.0]);
    assert!((a.apply(&x) - &dense * &x).norm() < 1e-12);
    assert_eq!(LinearOperator::dim(&a), 4);
  }

  #[test]
  fn identity_applies_unchanged() {
    let id = Identity::new(3);
    let r = Vector::from_column_slice(&[2.0, -1.0, 7.0]);
    assert_eq!(id.apply(&r), r);
  }

  /// Totality at the degenerate boundary: order zero is a defined, trivial op.
  #[test]
  fn identity_is_total_at_zero() {
    let id = Identity::new(0);
    assert_eq!(id.apply(&Vector::zeros(0)), Vector::zeros(0));
  }

  /// On a diagonal operator Jacobi is the exact inverse: $B A = I$.
  #[test]
  fn jacobi_inverts_a_diagonal_operator() {
    let a = csr(&DMatrix::from_diagonal(&Vector::from_column_slice(&[
      2.0, 5.0, 0.25, 8.0,
    ])));
    let b = Jacobi::new(&a);
    let x = Vector::from_column_slice(&[1.0, -3.0, 4.0, 2.0]);
    assert!((b.apply(&a.apply(&x)) - &x).norm() < 1e-12);
  }

  /// The law the `SelfAdjoint` marker promises: $angle.l B r, s angle.r =
  /// angle.l r, B s angle.r$. Verified on a full (non-diagonal) SPD operator,
  /// whose diagonal Jacobi reads.
  #[test]
  fn jacobi_is_self_adjoint() {
    let a = csr(&spd_from_spectrum(&[1.0, 2.0, 4.0, 7.0, 9.0]));
    let b = Jacobi::new(&a);
    let r = Vector::from_column_slice(&[1.0, -2.0, 3.0, 0.5, -1.0]);
    let s = Vector::from_column_slice(&[4.0, 1.0, -1.0, 2.0, 3.0]);
    assert!((b.apply(&r).dot(&s) - r.dot(&b.apply(&s))).abs() < 1e-12);
  }

  /// Stationary Jacobi iteration converges to the true solution on a
  /// diagonally dominant SPD system, at a rate set by $rho(I - D^(-1) A)$.
  #[test]
  fn stationary_converges_to_the_solution() {
    let dense = tridiag(8, 4.0, 1.0);
    let a = csr(&dense);
    let x_true = Vector::from_fn(8, |i, _| (i as f64 - 3.5).sin());
    let b = &dense * &x_true;

    let (x, report) = stationary::solve(&a, &Jacobi::new(&a), &b, StopCriterion::rtol(1e-10));
    assert!(report.converged);
    assert!((x - x_true).norm() < 1e-8);

    // The iteration count is governed by the spectral radius, not free: the
    // predicted geometric rate bounds it (with slack for the 2-norm transient).
    let n = dense.nrows();
    let dinv = DMatrix::from_diagonal(&dense.diagonal().map(|d| 1.0 / d));
    let rho = (DMatrix::identity(n, n) - dinv * &dense)
      .complex_eigenvalues()
      .iter()
      .map(|c| c.norm())
      .fold(0.0, f64::max);
    let predicted = (1e-10_f64.ln() / rho.ln()).ceil() as usize;
    assert!(rho < 1.0 && report.iters <= 3 * predicted + 10);
  }

  /// A fixed number of Jacobi sweeps is itself self-adjoint --- the promise the
  /// `SelfAdjoint for Stationary` impl makes, and the basis of nesting it inside
  /// a Krylov method.
  #[test]
  fn stationary_sweeps_are_self_adjoint() {
    let a = csr(&tridiag(6, 4.0, 1.0));
    let sweeps = Stationary::new(&a, Jacobi::new(&a), 3);
    let r = Vector::from_fn(6, |i, _| (i as f64).cos());
    let s = Vector::from_fn(6, |i, _| (2.0 * i as f64 + 1.0).sin());
    assert!((sweeps.apply(&r).dot(&s) - r.dot(&sweeps.apply(&s))).abs() < 1e-12);
  }

  /// CG's defining theorem: on an $n times n$ SPD system it reaches the exact
  /// solution in at most $n$ steps. Swept over orders, with the degenerate
  /// $n = 0, 1$ included so totality holds at the boundary. The spectrum is
  /// pinned (distinct eigenvalues), since finite termination degrades under
  /// ill-conditioning in floating point.
  #[test]
  fn cg_terminates_in_at_most_n_steps() {
    for n in 0..=8 {
      let eigs: Vec<f64> = (0..n).map(|k| 1.0 + k as f64).collect();
      let dense = spd_from_spectrum(&eigs);
      let a = csr(&dense);
      let x_true = Vector::from_fn(n, |i, _| (i as f64 + 1.0).ln());
      let b = &dense * &x_true;

      let stop = StopCriterion {
        rtol: 1e-10,
        max_iters: n.max(1),
      };
      let (x, report) = krylov::cg(&a, &Identity::new(n), &b, stop);
      assert!(report.converged, "n = {n} did not converge in {n} steps");
      assert!(report.iters <= n);
      if n > 0 {
        assert!((x - x_true).norm() < 1e-7, "n = {n}");
      }
    }
  }

  /// Preconditioning changes the path, never the fixed point: Jacobi-CG reaches
  /// the same solution as unpreconditioned CG.
  #[test]
  fn preconditioning_preserves_the_solution() {
    let dense = tridiag(20, 4.0, 1.0);
    let a = csr(&dense);
    let x_true = Vector::from_fn(20, |i, _| ((i * i) as f64).cos());
    let b = &dense * &x_true;
    let stop = StopCriterion::rtol(1e-12);

    let (x_plain, _) = krylov::cg(&a, &Identity::new(20), &b, stop);
    let (x_jacobi, _) = krylov::cg(&a, &Jacobi::new(&a), &b, stop);
    assert!((&x_plain - &x_true).norm() < 1e-9);
    assert!((&x_jacobi - &x_true).norm() < 1e-9);
    assert!((x_plain - x_jacobi).norm() < 1e-9);
  }

  /// The composition that justifies the whole trait algebra: a consumer
  /// (`Stationary`) used as an implementor, preconditioning another consumer
  /// (`cg`). CG preconditioned by two Jacobi sweeps solves the system.
  #[test]
  fn cg_preconditioned_by_stationary_sweeps() {
    let dense = tridiag(20, 4.0, 1.0);
    let a = csr(&dense);
    let x_true = Vector::from_fn(20, |i, _| (i as f64 - 10.0).tanh());
    let b = &dense * &x_true;

    let sweeps = Stationary::new(&a, Jacobi::new(&a), 2);
    let (x, report) = krylov::cg(&a, &sweeps, &b, StopCriterion::rtol(1e-10));
    assert!(report.converged);
    assert!((x - x_true).norm() < 1e-7);
  }

  /// MINRES solves a symmetric *indefinite* system --- the case CG cannot ---
  /// reproducing the direct solve, swept over orders including the degenerate
  /// $n = 0, 1$.
  #[test]
  fn minres_solves_symmetric_indefinite_systems() {
    use crate::testutil::symmetric_from_spectrum;
    for n in 0..=8 {
      // A mixed-sign spectrum bounded away from zero: symmetric, indefinite,
      // nonsingular. Magnitudes 1, 1, 2, 2, ... with alternating sign.
      let eigs: Vec<f64> = (0..n)
        .map(|k| (k / 2 + 1) as f64 * if k % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
      let dense = symmetric_from_spectrum(&eigs);
      let a = csr(&dense);
      let b = Vector::from_fn(n, |i, _| (i as f64 + 1.0).sqrt());

      let (x, report) = krylov::minres(&a, &Identity::new(n), &b, StopCriterion::rtol(1e-11));
      assert!(report.converged, "n = {n} did not converge");
      if n > 0 {
        assert!((x - dense_solve(&dense, &b)).norm() < 1e-7, "n = {n}");
      }
    }
  }

  /// On an SPD system MINRES and CG reach the same solution: MINRES is the
  /// generalization, agreeing where CG applies.
  #[test]
  fn minres_agrees_with_cg_on_spd() {
    let dense = tridiag(25, 4.0, 1.0);
    let a = csr(&dense);
    let x_true = Vector::from_fn(25, |i, _| (i as f64).sin());
    let b = &dense * &x_true;
    let stop = StopCriterion::rtol(1e-12);

    let (x_min, report) = krylov::minres(&a, &Jacobi::new(&a), &b, stop);
    assert!(report.converged);
    assert!((&x_min - &x_true).norm() < 1e-8);

    let (x_cg, _) = krylov::cg(&a, &Jacobi::new(&a), &b, stop);
    assert!((x_min - x_cg).norm() < 1e-7);
  }

  /// A block-diagonal preconditioner applies each block's inverse to its own
  /// slice: for a block-diagonal operator with exact (Jacobi-on-diagonal)
  /// blocks it is the exact inverse, so preconditioned CG converges in one step.
  #[test]
  fn block_diagonal_of_exact_blocks_is_exact() {
    // A block-diagonal operator, each block a distinct diagonal matrix.
    let sizes = [3usize, 4, 2];
    let n: usize = sizes.iter().sum();
    let diag = DMatrix::from_diagonal(&Vector::from_fn(n, |i, _| 1.0 + (i % 6) as f64));
    let a = csr(&diag);

    let mut blocks = Vec::new();
    let mut off = 0;
    for &d in &sizes {
      let sub = csr(&diag.view((off, off), (d, d)).into_owned());
      blocks.push(Jacobi::new(&sub));
      off += d;
    }
    let precond = BlockDiagonal::new(blocks);
    assert_eq!(precond.dim(), n);

    let b = Vector::from_fn(n, |i, _| (i as f64 - 4.0).cos());
    let (x, report) = krylov::cg(&a, &precond, &b, StopCriterion::rtol(1e-12));
    assert!(
      report.converged && report.iters <= 1,
      "iters = {}",
      report.iters
    );
    assert!((x - dense_solve(&diag, &b)).norm() < 1e-9);
  }

  /// A dense direct inverse, test-only coarse solver: the exact $A^(-1)$ at the
  /// bottom of a V-cycle, standing in for the faer factorization the FEEC
  /// consumer supplies. Self-adjoint for a symmetric operator.
  struct DenseInverse {
    inv: DMatrix<f64>,
  }
  impl DenseInverse {
    fn new(a: &DMatrix<f64>) -> Self {
      Self {
        inv: a.clone().try_inverse().expect("nonsingular"),
      }
    }
  }
  impl ApproxInverse for DenseInverse {
    type Space = Vector;
    fn dim(&self) -> usize {
      self.inv.nrows()
    }
    fn apply(&self, r: &Vector) -> Vector {
      &self.inv * r
    }
  }
  impl SelfAdjoint for DenseInverse {}

  /// The 1D Dirichlet Laplacian $"tridiag"(-1, 2, -1)$ on `n` interior points,
  /// scaled by $h^(-2) = (n+1)^2$: the model second-order SPD operator whose
  /// condition number grows like $h^(-2)$, so an $h$-independent iteration count
  /// is a nontrivial claim.
  fn laplacian_1d(n: usize) -> DMatrix<f64> {
    let h2_inv = ((n + 1) * (n + 1)) as f64;
    DMatrix::from_fn(n, n, |i, j| {
      if i == j {
        2.0 * h2_inv
      } else if i.abs_diff(j) == 1 {
        -h2_inv
      } else {
        0.0
      }
    })
  }

  /// Linear-interpolation prolongation from a coarse grid of `2^level - 1`
  /// interior points to the fine grid of `2^(level+1) - 1`: each coarse point
  /// maps to the fine point at the same location (weight 1) and its two fine
  /// neighbors (weight 1/2). Its transpose is full-weighting restriction.
  fn interpolation_1d(coarse: usize) -> CsrMatrix {
    let fine = 2 * coarse + 1;
    let mut coo = nas::CooMatrix::new(fine, coarse);
    for jc in 0..coarse {
      let jf = 2 * jc + 1; // coarse point jc sits at fine index 2 jc + 1
      coo.push(jf, jc, 1.0);
      coo.push(jf - 1, jc, 0.5);
      coo.push(jf + 1, jc, 0.5);
    }
    CsrMatrix::from(&coo)
  }

  /// Build a V-cycle for the 1D Laplacian tower of `levels` grids, Galerkin
  /// coarse operators $A_c = P^T A P$, damped-Jacobi smoothing. Returns the
  /// cycle and the finest operator.
  fn poisson_vcycle(
    levels: usize,
    sweeps: usize,
  ) -> (VCycle<Jacobi, DenseInverse>, CsrMatrix, DMatrix<f64>) {
    // Finest grid has 2^levels - 1 points; coarsen by exact bisection.
    let fine_pts = (1 << levels) - 1;
    let a_fine_dense = laplacian_1d(fine_pts);
    let a_fine = csr(&a_fine_dense);

    let mut ops = vec![a_fine.clone()];
    let mut prolongs = Vec::new();
    for _ in 1..levels {
      let coarse_pts = (ops.last().unwrap().nrows() - 1) / 2;
      let p = interpolation_1d(coarse_pts);
      let pt = p.transpose();
      let a_coarse = &pt * &(ops.last().unwrap() * &p);
      prolongs.push(p);
      ops.push(a_coarse);
    }

    let level_structs: Vec<Level<Jacobi>> = (0..levels - 1)
      .map(|i| {
        let p = prolongs[i].clone();
        let r = p.transpose();
        Level::new(ops[i].clone(), Jacobi::weighted(&ops[i], 2.0 / 3.0), p, r)
      })
      .collect();

    let coarsest = &ops[levels - 1];
    let coarse = DenseInverse::new(&DMatrix::from(coarsest));
    let cycle = VCycle::symmetric(level_structs, coarse, sweeps);
    (cycle, a_fine, a_fine_dense)
  }

  /// The V-cycle used as a standalone stationary iteration contracts the error at
  /// a rate bounded well below one, and that rate is essentially independent of
  /// the mesh --- the property that distinguishes multigrid from a one-level
  /// smoother. Measured as the asymptotic residual reduction factor on two grids
  /// an octave apart.
  #[test]
  fn vcycle_contracts_uniformly_in_h() {
    let factor = |levels: usize| -> f64 {
      let (cycle, a, _) = poisson_vcycle(levels, 2);
      let n = a.nrows();
      let x_true = Vector::from_fn(n, |i, _| (i as f64 + 1.0).sin());
      let b = &a * &x_true;
      // Stationary V-cycle iteration; read the reduction factor once transients
      // have died out.
      let mut x = Vector::zeros(n);
      let mut prev = f64::INFINITY;
      let mut factor = 0.0;
      for _ in 0..30 {
        x += cycle.apply(&(&b - &a * &x));
        let res = (&b - &a * &x).norm();
        factor = res / prev;
        prev = res;
        if res < 1e-11 * b.norm() {
          break;
        }
      }
      factor
    };
    let coarse = factor(4); // 15 points
    let fine = factor(7); //  127 points
    assert!(coarse < 0.4, "two-grid rate {coarse} not a contraction");
    assert!(fine < 0.4, "finer rate {fine} not a contraction");
    // h-independence: the rate must not degrade as the mesh is refined.
    assert!(
      fine < coarse + 0.1,
      "rate degraded under refinement: {coarse} -> {fine}"
    );
  }

  /// A V-cycle preconditions CG: it is self-adjoint, so the bound accepts it, and
  /// MG-CG reaches the same solution as plain CG in far fewer iterations, that
  /// count staying bounded as the mesh refines.
  #[test]
  fn vcycle_preconditioned_cg_is_mesh_independent() {
    let solve = |levels: usize| -> (usize, usize) {
      let (cycle, a, a_dense) = poisson_vcycle(levels, 2);
      let n = a.nrows();
      let x_true = Vector::from_fn(n, |i, _| (i as f64 - 3.0).tanh());
      let b = &a * &x_true;
      let stop = StopCriterion::rtol(1e-10);
      let (x_mg, rep_mg) = krylov::cg(&a, &cycle, &b, stop);
      let (_, rep_plain) = krylov::cg(&a, &Identity::new(n), &b, stop);
      assert!(rep_mg.converged);
      assert!(
        (x_mg - dense_solve(&a_dense, &b)).norm() < 1e-7,
        "MG-CG solution wrong at levels = {levels}"
      );
      (rep_mg.iters, rep_plain.iters)
    };
    let (mg_coarse, plain_coarse) = solve(4);
    let (mg_fine, plain_fine) = solve(7);
    // MG-CG iterations stay flat while plain CG grows with the condition number.
    assert!(mg_fine <= mg_coarse + 2, "{mg_coarse} -> {mg_fine} grew");
    assert!(
      plain_fine > 2 * plain_coarse,
      "plain CG should grow with h: {plain_coarse} -> {plain_fine}"
    );
    assert!(mg_fine < plain_fine / 4, "MG not beating plain CG");
  }

  /// Block-diagonal is self-adjoint when its blocks are, so it may precondition
  /// CG/MINRES exactly when they may.
  #[test]
  fn block_diagonal_is_self_adjoint_from_blocks() {
    let a1 = csr(&spd_from_spectrum(&[1.0, 2.0, 4.0]));
    let a2 = csr(&spd_from_spectrum(&[3.0, 5.0]));
    let precond = BlockDiagonal::new(vec![Jacobi::new(&a1), Jacobi::new(&a2)]);
    let r = Vector::from_fn(5, |i, _| (i as f64 + 1.0).ln());
    let s = Vector::from_fn(5, |i, _| (2.0 * i as f64).cos());
    assert!((precond.apply(&r).dot(&s) - r.dot(&precond.apply(&s))).abs() < 1e-12);
  }
}

#[cfg(test)]
mod space {
  use crate::{
    ApproxInverse, InnerProductSpace, LinearOperator, SelfAdjoint, StopCriterion, Vector,
    krylov::cg, testutil::spd_from_spectrum,
  };

  /// A realization of the space sharing no code with nalgebra: a plain `Vec`
  /// and hand-written arithmetic.
  ///
  /// The point of the second instance is that it is a *second* one. If the
  /// Krylov methods still reach the same iterate here, they read nothing about
  /// their vectors beyond [`InnerProductSpace`], which is what lets the same
  /// method run on vectors that never enter host memory.
  #[derive(Clone, Debug)]
  struct Coords(Vec<f64>);

  impl InnerProductSpace for Coords {
    fn zeros_like(&self) -> Self {
      Coords(vec![0.0; self.0.len()])
    }
    fn dot(&self, other: &Self) -> f64 {
      self.0.iter().zip(&other.0).map(|(a, b)| a * b).sum()
    }
    fn scale(&mut self, alpha: f64) {
      self.0.iter_mut().for_each(|y| *y *= alpha);
    }
    fn add_scaled(&mut self, alpha: f64, x: &Self) {
      for (y, x) in self.0.iter_mut().zip(&x.0) {
        *y += alpha * x;
      }
    }
  }

  /// A dense operator over [`Coords`], row-major, applied by hand.
  struct Dense {
    rows: Vec<Vec<f64>>,
  }
  impl LinearOperator for Dense {
    type Space = Coords;
    fn dim(&self) -> usize {
      self.rows.len()
    }
    fn apply(&self, x: &Coords) -> Coords {
      Coords(
        self
          .rows
          .iter()
          .map(|row| row.iter().zip(&x.0).map(|(a, b)| a * b).sum())
          .collect(),
      )
    }
  }

  struct Unpreconditioned(usize);
  impl ApproxInverse for Unpreconditioned {
    type Space = Coords;
    fn dim(&self) -> usize {
      self.0
    }
    fn apply(&self, r: &Coords) -> Coords {
      r.clone()
    }
  }
  impl SelfAdjoint for Unpreconditioned {}

  /// The same system solved in two unrelated realizations of the space agrees,
  /// iterate for iterate.
  ///
  /// Not merely to the tolerance: conjugate gradients is a deterministic
  /// recurrence in the inner products alone, so two spaces that agree on those
  /// must produce the same iterates, and the iteration counts must match
  /// exactly. A method that peeked at an entry would have no reason to.
  #[test]
  fn the_krylov_methods_read_nothing_but_the_space() {
    let dense = spd_from_spectrum(&[1.0, 2.0, 3.5, 6.0, 11.0]);
    let n = dense.nrows();
    let rhs: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sqrt()).collect();

    let (host, host_report) = cg(
      &crate::testutil::csr(&dense),
      &crate::Identity::new(n),
      &Vector::from_column_slice(&rhs),
      StopCriterion::rtol(1e-12),
    );

    let op = Dense {
      rows: (0..n)
        .map(|i| dense.row(i).iter().copied().collect())
        .collect(),
    };
    let (coords, coords_report) = cg(
      &op,
      &Unpreconditioned(n),
      &Coords(rhs),
      StopCriterion::rtol(1e-12),
    );

    assert_eq!(host_report.iters, coords_report.iters);
    for (h, c) in host.iter().zip(&coords.0) {
      assert!((h - c).abs() < 1e-12, "{h} vs {c}");
    }
  }
}
