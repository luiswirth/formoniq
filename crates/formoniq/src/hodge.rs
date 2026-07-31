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

/// The Hodge-Laplace differential complex around grade $k$, as the four
/// matrices a mixed problem is built from, on any [`HilbertComplex`], so the
/// trait alone decides natural (full complex) versus essential (relative
/// complex) boundary conditions.
///
/// The fields are what the complex supplies, one call each; the methods are the
/// couplings the saddle point pairs them into. Nothing here assembles.
///
/// The $omega = dif u in Lambda^(k+1)$ space is present in the mathematics and
/// absent from the data, deliberately. It reaches the block system only through
/// $codif_dif$, which the complex produces from its element form without ever
/// building $M_(k+1)$, so holding the up-mass here to contract it away would
/// assemble the largest skeleton in reach for a matrix no problem asks for: at
/// grade $0$ in 3D, the mass on the edges to make an operator on the vertices.
///
/// The two degenerate grades need no case of their own: the complex is total in
/// grade, so at $k = 0$ the $sigma$ space is empty and at $k = n$ the $omega$
/// space is, and the blocks come out correctly shaped from the same
/// expressions.
pub struct HodgeBlocks {
  pub n_sigma: usize,
  pub n_u: usize,
  /// $M_(k-1)$, the mass on the $sigma = delta u$ space.
  pub mass_sigma: CsrMatrix,
  /// $M_k$, the mass on the $u$ space.
  pub mass_u: CsrMatrix,
  /// $dif: Lambda^(k-1) -> Lambda^k$, the differential on the $sigma$ space,
  /// shape $n_u times n_sigma$.
  pub dif_sigma: CsrMatrix,
  /// $(D^k)^T M_(k+1) D^k$, the Galerkin matrix of $(dif u, dif v)$, shape
  /// $n_u^2$. The up-Laplacian $delta dif$, zero at top grade where $dif u = 0$.
  ///
  /// Named for the form it is rather than for stiffness, which is a word from
  /// structural mechanics and says nothing here: it is
  /// [`HilbertComplex::codif_dif`], and one object carries one name.
  pub codif_dif: CsrMatrix,
}
impl HodgeBlocks {
  pub fn compute<C: HilbertComplex>(complex: &C, grade: ExteriorGrade) -> Self {
    assert!(grade <= complex.dim());
    Self {
      n_sigma: complex.ndofs(grade - 1),
      n_u: complex.ndofs(grade),
      mass_sigma: CsrMatrix::from(&complex.mass(grade - 1)),
      mass_u: CsrMatrix::from(&complex.mass(grade)),
      dif_sigma: complex.dif(grade - 1),
      codif_dif: CsrMatrix::from(&complex.codif_dif(grade)),
    }
  }

  /// The weak codifferential $delta$ of the $u$ space, $(D^(k-1))^T M_k$, shape
  /// $n_sigma times n_u$: the $sigma <- u$ block, characterized by
  /// $angle.l delta u, tau angle.r = angle.l u, dif tau angle.r$.
  ///
  /// The dual of [`Self::dif_sigma`], and the pair is why they are named for
  /// the space each acts on rather than for a direction: $dif$ takes $sigma$ up
  /// into $Lambda^k$, $delta$ takes $u$ back down.
  ///
  /// The opposite block $u <- sigma$ is this transposed, and is not a second
  /// product: the mass is symmetric on every signature, so
  /// $(D^T M)^T = M D$. That adjointness is exactly what makes the saddle point
  /// symmetric, so a caller transposing states it where a second method would
  /// reproduce it.
  pub fn codif_u(&self) -> CsrMatrix {
    self.dif_sigma.transpose() * &self.mass_u
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
    DirectInverse::new(self.mass_sigma.clone()).apply(&(self.codif_u() * u))
  }

  /// The mixed Hodge-Laplacian $mat(M_(k-1), -(D^(k-1))^T M_k; M_k D^(k-1), K)$
  /// on $(sigma, u)$: the saddle point of the first-order system
  /// $sigma = delta u$, $dif sigma + delta dif u = f$.
  pub fn mixed_hodge_laplacian(&self) -> CooMatrix {
    let coo = |m: &CsrMatrix| CooMatrix::from(m);
    CooMatrix::block(&[
      &[&coo(&self.mass_sigma), &coo(&self.codif_u()).neg()],
      &[&coo(&self.codif_u().transpose()), &coo(&self.codif_dif)],
    ])
  }
}
