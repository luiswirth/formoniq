//! The Hodge-Laplacian of a discrete Hilbert complex, in its assembled blocks.
//!
//! Every problem the engine poses around a grade $k$, elliptic, parabolic,
//! hyperbolic and Dirac alike, is built from the same three masses and two
//! coboundaries, so they are one object here rather than a structure each
//! problem re-derives. What varies between the problems is which blocks they
//! arrange and into what system, not what the blocks are.

use crate::whitney_complex::HilbertComplex;

use multialgebra::ExteriorGrade;
use simplicial::linalg::{CooMatrix, CooMatrixExt, CsrMatrix, Vector};

use crate::linalg::DirectInverse;
use iterative::ApproxInverse;

/// The factored per-grade blocks of the Hodge-Laplace differential complex
/// around grade $k$, the mass matrices $M_(k-1), M_k, M_(k+1)$ and the
/// metric-free coboundaries $D^(k-1), D^k$, assembled on any
/// [`HilbertComplex`], so the trait alone decides natural (full complex) versus
/// essential (relative complex) boundary conditions.
///
/// These are the pieces the mixed problems build their block systems from: the
/// down-coupling $sigma = delta u in Lambda^(k-1)$ and the up-coupling
/// $omega = dif u in Lambda^(k+1)$. The two degenerate grades ($k = 0$ has no
/// $sigma$ space, $k = n$ has no $omega$ space) need no case of their own: the
/// complex is total in grade, so the neighbouring space is empty there and its
/// blocks come out correctly shaped from the same expressions.
pub struct HodgeBlocks {
  pub n_sigma: usize,
  pub n_u: usize,
  pub n_omega: usize,
  pub mass_sigma: CsrMatrix,
  pub mass_u: CsrMatrix,
  pub mass_omega: CsrMatrix,
  /// $D^(k-1): Lambda^(k-1) -> Lambda^k$, shape $n_u times n_sigma$.
  pub dif_dn: CsrMatrix,
  /// $D^k: Lambda^k -> Lambda^(k+1)$, shape $n_omega times n_u$.
  pub dif_up: CsrMatrix,
}
impl HodgeBlocks {
  pub fn compute<C: HilbertComplex>(complex: &C, grade: ExteriorGrade) -> Self {
    assert!(grade <= complex.dim());
    Self {
      n_sigma: complex.ndofs(grade - 1),
      n_u: complex.ndofs(grade),
      n_omega: complex.ndofs(grade + 1),
      mass_sigma: CsrMatrix::from(&complex.mass(grade - 1)),
      mass_u: CsrMatrix::from(&complex.mass(grade)),
      mass_omega: CsrMatrix::from(&complex.mass(grade + 1)),
      dif_dn: complex.dif(grade - 1),
      dif_up: complex.dif(grade),
    }
  }

  /// The weak codifferential coupling $angle.l dif tau, u angle.r$ as a matrix
  /// $(D^(k-1))^T M_k$, shape $n_sigma times n_u$: the $sigma <- u$ block.
  pub fn codif_dn(&self) -> CsrMatrix {
    self.dif_dn.transpose() * &self.mass_u
  }

  /// The weak exterior-derivative coupling $angle.l dif sigma, v angle.r$ as a
  /// matrix $M_k D^(k-1)$, shape $n_u times n_sigma$: the $u <- sigma$ block.
  pub fn dif_sigma(&self) -> CsrMatrix {
    &self.mass_u * &self.dif_dn
  }

  /// The weak coupling $angle.l omega, dif v angle.r$ as a matrix
  /// $(D^k)^T M_(k+1)$, shape $n_u times n_omega$: the $u <- omega$ block.
  pub fn codif_up(&self) -> CsrMatrix {
    self.dif_up.transpose() * &self.mass_omega
  }

  /// The weak coupling $angle.l dif mu, phi angle.r$ as a matrix
  /// $M_(k+1) D^k$, shape $n_omega times n_u$: the $omega <- u$ block.
  pub fn dif_omega(&self) -> CsrMatrix {
    &self.mass_omega * &self.dif_up
  }

  /// The up-Laplacian stiffness $K = (D^k)^T M_(k+1) D^k$ ($delta dif$), shape
  /// $n_u^2$. Zero at top grade, where $dif u = 0$.
  pub fn stiff(&self) -> CsrMatrix {
    self.dif_up.transpose() * &self.mass_omega * &self.dif_up
  }

  /// The codifferential $sigma = delta u$ of a coefficient vector, the solution
  /// of the algebraic relation $M_(k-1) sigma = (D^(k-1))^T M_k u$ that slaves
  /// $sigma$ to $u$ in every mixed system here.
  ///
  /// Total in grade: at grade $0$ the $sigma$ space is trivial and the empty
  /// vector is its only element, so the mass solve is skipped rather than run
  /// on an empty system. Not total over signature: the solve is the SPD direct
  /// one, so it wants a Riemannian geometry, where an indefinite $M_(k-1)$
  /// needs the LU its caller must then choose.
  pub fn codif(&self, u: &Vector) -> Vector {
    if self.n_sigma == 0 {
      return Vector::zeros(0);
    }
    DirectInverse::new(self.mass_sigma.clone()).apply(&(self.codif_dn() * u))
  }

  /// The mixed Hodge-Laplacian $mat(M_(k-1), -(D^(k-1))^T M_k; M_k D^(k-1), K)$
  /// on $(sigma, u)$: the saddle point of the first-order system
  /// $sigma = delta u$, $dif sigma + delta dif u = f$.
  pub fn mixed_hodge_laplacian(&self) -> CooMatrix {
    let coo = |m: &CsrMatrix| CooMatrix::from(m);
    CooMatrix::block(&[
      &[&coo(&self.mass_sigma), &coo(&self.codif_dn()).neg()],
      &[&coo(&self.dif_sigma()), &coo(&self.stiff())],
    ])
  }
}
