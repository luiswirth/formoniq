//! The discrete harmonic space, computed from integral cohomology rather than
//! from an eigensolve.
//!
//! Hodge theory says the harmonic space $cal(H)^k$ is isomorphic to the
//! cohomology $H^k$, and the isomorphism is explicit: a class is represented by
//! a cocycle $z$, and the harmonic representative is the one of *least* $L^2$
//! norm in the affine set $z + "im" dif^(k-1)$,
//!
//! $ h = z - D p, quad p = arg min_q norm(z - D q)_(M_k), $
//!
//! so $p$ solves the normal equations $(D^T M_k D) p = D^T M_k z$. Then
//! $h perp_(M_k) "im" dif^(k-1)$ by construction and $dif h = dif z = 0$
//! because $z$ is a cocycle, which is exactly discrete harmonicity.
//!
//! Two things this buys over an eigensolve of the Hodge-Laplace pencil near
//! $0$. First, $dif h = 0$ holds *exactly*: an integral cocycle and a
//! $plus.minus 1$ incidence multiply to an exact zero, where an eigensolver
//! resolves a cluster of $b_k$ near-zero eigenvalues only to its own tolerance.
//! Second, each basis vector is tied to a specific integral cohomology class,
//! hence to a specific hole, reproducibly and stably under refinement, where an
//! eigensolver returns an arbitrary rotation within the null space.
//!
//! The projection alone leaves the basis of $cal(H)^k$ only defined up to
//! $"GL"(b_k)$, since the cocycles it starts from are whichever ones the
//! elimination order produced, so a representative may still wrap a combination
//! of holes. The basis is pinned by its **periods**: pairing against the
//! Kronecker-dual cycles gives $P_(i j) = integral_(z_j) h^i$, nonsingular over
//! $QQ$, and $H |-> H P^(-T)$ is the basis with
//!
//! $ integral_(z_j) h^i = delta^i_j, $
//!
//! so basis vector $i$ has unit period around hole $i$ and none around any
//! other. The periods are computed on the *cocycles*, over $ZZ$ and hence
//! exactly, because a period does not see the projection: $h = z - D p$ and
//! $chevron.l D p, z_j chevron.r = chevron.l p, partial z_j chevron.r = 0$ on a cycle, so
//! $h$ and $z$ have the same periods. Pinning the basis this way inherits the
//! labelling of the cycles: it makes the correspondence to the holes explicit,
//! it does not choose which hole is which.
//!
//! $A = D^T M_k D$ is singular, its kernel being $ker dif^(k-1)$, but the
//! system is consistent by construction and the residual $h = z - D p$ does not
//! depend on which solution $p$ is taken, so nothing has to be gauge-fixed and
//! CG on the consistent semidefinite system suffices.
//!
//! This is the Riemannian path. On a Lorentzian geometry the $L^2$ pairing is
//! indefinite, the minimization is not well posed, and there is no orthogonal
//! projection to take; [`harmonics`] returns `None` there, read off a
//! factorization rather than off the metric, as
//! `mixed_block_preconditioner` does.

use crate::{linalg::bilinear_form_sparse, whitney_complex::HilbertComplex};

use derham::{Chain, Cochain, pairing};
use iterative::{Identity, StopCriterion, krylov::cg};
use multialgebra::ExteriorGrade;
use simplicial::linalg::Matrix;

/// The two readings of one harmonic space, as the columns of two matrices
/// spanning it.
///
/// They are different bases of the *same* subspace and both are wanted:
/// [`Self::integral`] carries the correspondence to the integral cohomology
/// classes, hence to individual holes, which is what a visualization means by a
/// harmonic form; [`Self::orthonormal`] is $M_k$-orthonormal, which is what the
/// mixed saddle point assumes of its harmonic block, its preconditioner using
/// the identity there. Neither substitutes for the other.
pub struct Harmonics {
  /// The harmonic representatives of the integral cohomology basis dual to
  /// [`HilbertComplex::integral_cycles`], in the order that gives:
  /// $integral_(z_j) h^i = delta^i_j$, so column $i$ carries unit period
  /// around cycle $i$ and none around the others.
  pub integral: Matrix,
  /// The $M_k$-orthonormalization of [`Self::integral`],
  /// $H L^(-T)$ for the Cholesky factor $G = L L^T$ of the Gram matrix
  /// $G = H^T M_k H$.
  pub orthonormal: Matrix,
}

/// The change of basis $H |-> H P^(-T)$ making the harmonic representatives
/// dual to the cycles, $integral_(z_j) h^i = delta^i_j$.
///
/// The period matrix $P_(i j) = chevron.l z^i, z_j chevron.r$ is read off the
/// integral cocycles rather than the projected forms, exactly over $ZZ$: a
/// period is blind to the projection, since the coboundary subtracted pairs to
/// zero against a cycle.
///
/// `None` where $P$ is singular, which Kronecker duality excludes over $QQ$: a
/// harmonic space whose periods do not separate its own dual cycles has lost
/// the correspondence to the holes that is the point of this basis, and there
/// is nothing to return in place of it.
fn period_normalize(
  integral: Matrix,
  cocycles: &[Cochain<i64>],
  cycles: &[Chain<i64>],
) -> Option<Matrix> {
  if integral.ncols() == 0 {
    return Some(integral);
  }
  let periods = Matrix::from_fn(cocycles.len(), cycles.len(), |i, j| {
    pairing(&cocycles[i], &cycles[j]) as f64
  });
  Some(integral * periods.transpose().try_inverse()?)
}

/// A basis of the discrete harmonic space $cal(H)^k$, in both readings.
///
/// The [`Harmonics::integral`] reading is period-normalized against
/// [`HilbertComplex::integral_cycles`], so its columns correspond to the holes
/// one for one; [`Harmonics::orthonormal`] is derived from it and is
/// $M_k$-orthonormal whichever basis of the space it is handed, so the saddle
/// point's assumption is independent of that normalization.
///
/// `None` where the Gram matrix of the harmonic representatives is not positive
/// definite, i.e. on an indefinite ($L^2$-pseudo-)metric, where the projection
/// this rests on is not well posed. The caller falls back to an eigensolve.
/// Also `None` on a singular period matrix, which `period_normalize` says
/// cannot arise from a dual pair of bases.
pub fn harmonics<C: HilbertComplex>(
  complex: &C,
  grade: impl Into<ExteriorGrade>,
) -> Option<Harmonics> {
  let grade = grade.into();
  let ndofs = complex.ndofs(grade);
  let cocycles = complex.integral_cocycles(grade);

  let mass = complex.mass(grade);
  // $D = dif^(k-1)$, the coboundary *into* this grade. At grade $0$ it has no
  // columns, the normal equations are the empty system and $h = z$ with no
  // special case.
  let dif = complex.dif(grade - 1);
  let normal = &(dif.transpose() * &mass) * &dif;
  let precond = Identity::new(normal.nrows());

  let columns: Vec<_> = cocycles
    .iter()
    .map(|cocycle| {
      let z = cocycle.extend_scalars(|&c| c as f64).coeffs().clone();
      let rhs = dif.transpose() * (&mass * &z);
      let (p, _) = cg(&normal, &precond, &rhs, StopCriterion::rtol(1e-12));
      z - &dif * p
    })
    .collect();
  // A complex with no cohomology in this grade has an empty harmonic space,
  // whose one basis is the empty one. `from_columns` cannot infer its shape.
  let integral = if columns.is_empty() {
    Matrix::zeros(ndofs, 0)
  } else {
    Matrix::from_columns(&columns)
  };
  let integral = period_normalize(integral, &cocycles, &complex.integral_cycles(grade))?;

  let gram = Matrix::from_fn(integral.ncols(), integral.ncols(), |i, j| {
    bilinear_form_sparse(
      &mass,
      &integral.column(i).into_owned(),
      &integral.column(j).into_owned(),
    )
  });
  // The signature guard: a Cholesky exists exactly when the $L^2$ pairing is
  // definite on this space. The empty Gram is trivially so.
  let orthonormal = if gram.is_empty() {
    integral.clone()
  } else {
    let factor = gram.cholesky()?;
    factor
      .l()
      .solve_lower_triangular(&integral.transpose())?
      .transpose()
  };

  Some(Harmonics {
    integral,
    orthonormal,
  })
}
