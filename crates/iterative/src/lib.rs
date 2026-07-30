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

/// A linear operator $A: V -> V$, applied as $x |-> A x$, the only
/// thing a Krylov method asks of its system matrix.
///
/// Distinct from [`ApproxInverse`] by intent, not by shape: this is $A$: that is
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
/// alone it is a solver; wrapped in a Krylov method it is a preconditioner;
/// sitting on a level of a multigrid hierarchy it is a smoother. The exact
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

/// The solve of a zero right-hand side: $x = 0$, in no iterations.
///
/// The degenerate case every method here shares, and the one place it is
/// answered. A relative residual is measured against $norm(b)$, which is the
/// only quantity in the crate that a caller can make zero, so the answer is
/// read off rather than computed.
pub(crate) fn trivial_solve<S: InnerProductSpace>(b: &S) -> (S, Report) {
  (
    b.zeros_like(),
    Report {
      iters: 0,
      residual: 0.0,
      converged: true,
    },
  )
}

#[cfg(test)]
mod testutil {
  use crate::{ApproxInverse, CsrMatrix, SelfAdjoint, Vector};
  use na::DMatrix;

  /// A dense direct inverse: the exact $A^(-1)$, standing in for the
  /// factorization a consumer supplies at the bottom of a V-cycle or on an
  /// auxiliary space. Self-adjoint for a symmetric operator.
  pub struct DenseInverse {
    inv: DMatrix<f64>,
  }
  impl DenseInverse {
    pub fn new(a: &DMatrix<f64>) -> Self {
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

  /// A symmetric operator with a prescribed spectrum,
  /// $A = Q "diag"(lambda) Q^T$ with $Q$ a deterministic orthogonal factor.
  /// Positive definiteness is the caller's choice of $lambda$, which is also
  /// what separates the CG-shaped case from the MINRES-shaped one.
  ///
  /// Controlled conditioning: the finite-termination law degrades under an
  /// ill-conditioned random matrix, so the spectrum is pinned, not sampled.
  pub fn symmetric_from_spectrum(eigs: &[f64]) -> DMatrix<f64> {
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

  /// Direct dense solve, the reference an iterative method must reproduce.
  pub fn dense_solve(a: &DMatrix<f64>, b: &Vector) -> Vector {
    a.clone().lu().solve(b).expect("nonsingular")
  }
}
