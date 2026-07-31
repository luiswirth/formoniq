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
//! $angle.l D p, z_j angle.r = angle.l p, diff z_j angle.r = 0$ on a cycle, so
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
/// The period matrix $P_(i j) = angle.l z^i, z_j angle.r$ is read off the
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

#[cfg(test)]
mod test {
  use super::*;
  use crate::whitney_complex::WhitneyComplex;
  use regge::{
    lengths::mesh::MeshLengthsSq, mesher::cartesian::CartesianGrid, mesher::quotient::FlatQuotient,
  };
  use simplicial::{Dim, linalg::Vector, topology::complex::Complex};

  /// A cube (trivial cohomology in positive grade, and the relative complex
  /// dual to it) and a torus (every $b_k = binom(d, k)$ nonzero): the second is
  /// what keeps these laws from being statements about the empty basis.
  fn fixtures() -> Vec<(Complex, MeshLengthsSq)> {
    let mut meshes: Vec<_> = (1..=3)
      .map(Dim::from)
      .map(|dim| {
        let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
        let lengths = coords.to_edge_lengths_sq(&topology);
        (topology, lengths)
      })
      .collect();
    meshes.extend(
      (1..=2)
        .map(Dim::from)
        .map(|dim| FlatQuotient::unit_torus(dim, 3).triangulate()),
    );
    meshes
  }

  /// The laws every harmonic basis obeys, checked on one complex and grade.
  fn assert_harmonic<C: HilbertComplex>(complex: &C, grade: Dim) {
    let harmonics = harmonics(complex, grade).expect("Riemannian, so the projection is well posed");
    let mass = complex.mass(grade);
    let dif_prev = complex.dif(grade - 1);
    let dif = complex.dif(grade);

    // The dimension of the harmonic space is a topological invariant, taken
    // from cohomology rather than from an eigenvalue tolerance.
    assert_eq!(harmonics.integral.ncols(), complex.harmonic_dim(grade));
    assert_eq!(harmonics.orthonormal.ncols(), complex.harmonic_dim(grade));

    for h in harmonics.integral.column_iter() {
      let h = h.into_owned();
      let scale = h.norm().max(1.0);

      // Closed: $dif h = dif z = 0$, since $z$ is a cocycle and $dif$ kills the
      // coboundary that was subtracted.
      assert!((&dif * &h).norm() <= 1e-8 * scale, "grade={grade}");

      // Coclosed weakly: $h perp_(M_k) "im" dif^(k-1)$, the defining property
      // of the least-squares residual.
      let coclosed = dif_prev.transpose() * (&mass * &h);
      assert!(coclosed.norm() <= 1e-8 * scale, "grade={grade}");
    }

    // The integral reading is dual to the cycles: $integral_(z_j) h^i =
    // delta^i_j$, so a basis form wraps its own hole and no other. This is a
    // law with content, since the unnormalized basis fails it.
    let cycles = complex.integral_cycles(grade);
    assert_eq!(cycles.len(), complex.harmonic_dim(grade));
    for (i, h) in harmonics.integral.column_iter().enumerate() {
      let form = Cochain::new(grade, h.into_owned());
      for (j, cycle) in cycles.iter().enumerate() {
        let period = pairing(&form, &cycle.extend_scalars(|&c| c as f64));
        let expected = f64::from(u8::from(i == j));
        assert!((period - expected).abs() <= 1e-8, "grade={grade}");
      }
    }

    // The orthonormal reading is what the mixed saddle point assumes:
    // $H^T M_k H = I$.
    let gram = Matrix::from_fn(
      harmonics.orthonormal.ncols(),
      harmonics.orthonormal.ncols(),
      |i, j| {
        bilinear_form_sparse(
          &mass,
          &harmonics.orthonormal.column(i).into_owned(),
          &harmonics.orthonormal.column(j).into_owned(),
        )
      },
    );
    for i in 0..gram.nrows() {
      for j in 0..gram.ncols() {
        assert!((gram[(i, j)] - f64::from(u8::from(i == j))).abs() <= 1e-8);
      }
    }
  }

  /// The harmonic representative is closed, weakly coclosed, and there are
  /// $b_k$ of them: the discrete Hodge isomorphism, over dimensions and grades
  /// and over both boundary conditions.
  #[test]
  fn harmonics_are_harmonic() {
    for (topology, lengths) in fixtures() {
      let whitney = WhitneyComplex::new(&topology, &lengths);
      let relative = whitney.relative();
      for grade in topology.dim().range_inclusive() {
        assert_harmonic(&whitney, grade);
        assert_harmonic(&relative, grade);
      }
    }
  }

  /// The harmonic representative is the $L^2$-minimal element of its class:
  /// $norm(h)_M <= norm(z - D q)_M$ for every $q$, sampled here over the
  /// coordinate directions and their sum. This is the claim that makes the
  /// representative canonical within the class, and the one the projection is
  /// solving for.
  #[test]
  fn harmonic_representative_is_l2_minimal() {
    for (topology, lengths) in fixtures() {
      let whitney = WhitneyComplex::new(&topology, &lengths);
      for grade in topology.dim().range_inclusive() {
        let mass = whitney.mass(grade);
        let dif_prev = whitney.dif(grade - 1);
        let harmonics = harmonics(&whitney, grade).unwrap();

        for (cocycle, h) in whitney
          .integral_cocycles(grade)
          .iter()
          .zip(harmonics.integral.column_iter())
        {
          let z = cocycle.extend_scalars(|&c| c as f64).coeffs().clone();
          let norm_h = bilinear_form_sparse(&mass, &h.into_owned(), &h.into_owned());

          let ndofs_prev = whitney.ndofs(grade - 1);
          let candidates = (0..ndofs_prev)
            .map(|i| Vector::from_fn(ndofs_prev, |r, _| f64::from(u8::from(r == i))))
            .chain(std::iter::once(Vector::from_element(ndofs_prev, 1.0)));
          for q in candidates {
            let competitor = &z - &dif_prev * q;
            let norm_competitor = bilinear_form_sparse(&mass, &competitor, &competitor);
            assert!(norm_h <= norm_competitor + 1e-9, "grade={grade}");
          }
        }
      }
    }
  }
}
