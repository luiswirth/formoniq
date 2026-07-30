//! Integer simplicial cohomology of the complex.
//!
//! The cochain complex
//!
//! $dots.c ->^(dif^(k-1)) C^k ->^(dif^k) C^(k+1) ->^(dif^(k+1)) dots.c$
//!
//! is the chain complex of [`homology`](super::homology) with its arrows
//! reversed: $dif^k$ is the transpose of $diff_(k+1)$, one incidence read the
//! other way, exactly as [`chain`](super::chain) says of the two differentials.
//! So a cohomology class $H^k = ker dif^k slash "im" dif^(k-1)$ and a homology
//! class are the same subquotient of the same matrix, and both are
//! the shared `quotient_generators` routine.
//!
//! There is deliberately no cohomological Betti number. Over $ZZ$ the free
//! ranks of $H^k$ and $H_k$ agree by universal coefficients, so
//! [`Complex::betti_number`] is the rank of both and a second function would be
//! one datum in two places. Metric-free (invariant 5): cohomology is a function
//! of the incidence alone.
//!
//! The generators pair with the homology ones (Kronecker), and that pairing is
//! nonsingular over $QQ$: a cohomology generator measures the periods of the
//! cycles. It need not be unimodular here, because
//! `quotient_generators` returns a $QQ$-basis of the free part rather than a
//! $ZZ$-basis of the integral lattice.

use super::{
  chain::{Chain, Cochain},
  complex::Complex,
};
use crate::Dim;
use crate::linalg::exact::{IntegerMatrix, quotient_generators};

impl Complex {
  /// The integer coboundary $dif^k: C^k -> C^(k+1)$, as the transpose of
  /// $diff_(k+1)$.
  ///
  /// Total over every grade, inheriting the totality of
  /// [`Self::integral_boundary`]: off the range it is the map between zero
  /// modules.
  fn integral_coboundary(&self, grade: Dim) -> IntegerMatrix {
    self.integral_boundary(grade + 1).transpose()
  }

  /// Representative cocycles whose classes are a basis of the free part of
  /// $H^k (K; ZZ)$, one [`Cochain`] per Betti number $b_k$.
  ///
  /// Each generator is a k-cocycle, $dif^k z = 0$, exactly: the coefficients
  /// are integers and the incidence entries are $plus.minus 1$, so nothing here
  /// is closed only to a tolerance. Its class generates a $ZZ$-summand of
  /// $H^k$, and the $b_k$ classes are independent modulo coboundaries.
  ///
  /// The caveats of `quotient_generators` apply: representatives chosen by
  /// the elimination order, never minimizers, spanning the free part over $QQ$
  /// without necessarily generating its integral lattice.
  pub fn cohomology_generators(&self, grade: Dim) -> Vec<Cochain<i64>> {
    quotient_generators(
      &self.integral_coboundary(grade),
      &self.integral_coboundary(grade - 1),
    )
    .into_iter()
    .map(|cocycle| Cochain::from_vec(grade, cocycle))
    .collect()
  }

  /// Representative cocycles of a basis of the free part of the relative
  /// cohomology $H^k (K, diff K; ZZ)$, one per
  /// [`relative_betti_number`](Self::relative_betti_number).
  ///
  /// The relative cochains *are* the cochains vanishing on $diff K$, so the
  /// relative complex is the cochain complex on the interior simplices
  /// (the interior selection) and a class is written
  /// back out as a full-length cochain by extension by zero. That embedding is
  /// the natural one, not a padding convention.
  ///
  /// These are the cocycles of the essential-boundary-condition de Rham
  /// complex, whose harmonic space they represent.
  pub fn relative_cohomology_generators(&self, grade: Dim) -> Vec<Cochain<i64>> {
    let interior = self.interior_selection(grade);
    let outgoing = self
      .integral_coboundary(grade)
      .submatrix(&self.interior_selection(grade + 1), &interior);
    let incoming = self
      .integral_coboundary(grade - 1)
      .submatrix(&interior, &self.interior_selection(grade - 1));

    quotient_generators(&outgoing, &incoming)
      .into_iter()
      .map(|cocycle| Cochain::from_vec(grade, interior.scatter(&cocycle)))
      .collect()
  }
}

/// The Kronecker pairing matrix $P_(i j) = angle.l z^i, z_j angle.r$ of a set
/// of cochains against a set of chains.
pub fn kronecker_matrix(cocycles: &[Cochain<i64>], cycles: &[Chain<i64>]) -> Vec<Vec<i64>> {
  cocycles
    .iter()
    .map(|cocycle| {
      cycles
        .iter()
        .map(|cycle| super::chain::pairing(cocycle, cycle))
        .collect()
    })
    .collect()
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::linalg::exact::IntegerMatrix;
  use crate::topology::homology::test::{annulus, test_complexes, two_sphere};

  /// Whether a cochain is a cocycle: $dif^k z = 0$, over $ZZ$ hence exactly.
  fn is_cocycle(complex: &Complex, cochain: &Cochain<i64>) -> bool {
    cochain.dif(complex).coeffs().iter().all(|&c| c == 0)
  }

  /// Universal coefficients: the free rank of $H^k$ is the Betti number of
  /// $H_k$, in every grade.
  #[test]
  fn generators_count_matches_betti() {
    for complex in test_complexes() {
      for k in complex.dim().range_inclusive() {
        assert_eq!(
          complex.cohomology_generators(k).len(),
          complex.betti_number(k),
          "grade {k}"
        );
      }
    }
  }

  /// Every generator is a cocycle.
  #[test]
  fn generators_are_cocycles() {
    for complex in test_complexes() {
      for k in complex.dim().range_inclusive() {
        for generator in complex.cohomology_generators(k) {
          assert!(is_cocycle(&complex, &generator), "grade {k}");
        }
      }
    }
  }

  /// The generator classes are independent modulo coboundaries: appended to the
  /// columns of $dif^(k-1)$ they raise the rank by exactly $b_k$, so no
  /// generator, nor any combination, is itself a coboundary.
  #[test]
  fn generators_independent_modulo_coboundaries() {
    for complex in test_complexes() {
      for k in complex.dim().range_inclusive() {
        let coboundaries = complex.integral_coboundary(k - 1);
        let generators = complex.cohomology_generators(k);

        let mut triplets = coboundaries.triplets().to_vec();
        for (g, generator) in generators.iter().enumerate() {
          for (kidx, &coeff) in generator.support() {
            triplets.push((kidx, coboundaries.ncols() + g, coeff));
          }
        }
        let augmented = IntegerMatrix::new(
          coboundaries.nrows(),
          coboundaries.ncols() + generators.len(),
          triplets,
        );
        assert_eq!(
          augmented.rank(),
          coboundaries.rank() + complex.betti_number(k),
          "grade {k}"
        );
      }
    }
  }

  /// Kronecker duality: the pairing of the cohomology generators against the
  /// homology ones is nonsingular, so the two bases are dual up to the
  /// invertible matrix $P$. Nonsingular, not unimodular: neither side is
  /// guaranteed to be a $ZZ$-basis of its lattice.
  #[test]
  fn kronecker_pairing_is_nonsingular() {
    for complex in test_complexes() {
      for k in complex.dim().range_inclusive() {
        let cocycles = complex.cohomology_generators(k);
        let cycles = complex.homology_generators(k);
        let pairing = kronecker_matrix(&cocycles, &cycles);

        let b = complex.betti_number(k);
        let triplets = pairing
          .iter()
          .enumerate()
          .flat_map(|(i, row)| row.iter().enumerate().map(move |(j, &v)| (i, j, v)))
          .collect();
        assert_eq!(IntegerMatrix::new(b, b, triplets).rank(), b, "grade {k}");
      }
    }
  }

  /// Kronecker duality for the pair: the relative cohomology generators pair
  /// nonsingularly with the relative homology ones, which is what lets the
  /// relative harmonic basis be pinned by periods exactly as the absolute one
  /// is.
  #[test]
  fn relative_kronecker_pairing_is_nonsingular() {
    for complex in test_complexes() {
      for k in complex.dim().range_inclusive() {
        let cocycles = complex.relative_cohomology_generators(k);
        let cycles = complex.relative_homology_generators(k);
        let pairing = kronecker_matrix(&cocycles, &cycles);

        let b = complex.relative_betti_number(k);
        let triplets = pairing
          .iter()
          .enumerate()
          .flat_map(|(i, row)| row.iter().enumerate().map(move |(j, &v)| (i, j, v)))
          .collect();
        assert_eq!(IntegerMatrix::new(b, b, triplets).rank(), b, "grade {k}");
      }
    }
  }

  /// The relative generators match the relative Betti numbers and vanish on the
  /// boundary.
  #[test]
  fn relative_generators_count_and_support() {
    for complex in test_complexes() {
      for k in complex.dim().range_inclusive() {
        let generators = complex.relative_cohomology_generators(k);
        assert_eq!(
          generators.len(),
          complex.relative_betti_number(k),
          "grade {k}"
        );
        let interior = complex.interior_selection(k);
        for generator in &generators {
          assert!(is_cocycle(&complex, generator), "grade {k}");
          assert!(
            generator
              .support()
              .all(|(kidx, _)| interior.position(kidx).is_some()),
            "a relative cocycle must vanish on the boundary, grade {k}"
          );
        }
      }
    }
  }

  /// The annulus has one 1-dimensional cohomology class, the one measuring the
  /// winding around the hole: it pairs nontrivially with the loop that
  /// generates $H_1$.
  #[test]
  fn annulus_cocycle_measures_the_loop() {
    let complex = annulus();
    let grade = Dim::new(1);
    let cocycles = complex.cohomology_generators(grade);
    let cycles = complex.homology_generators(grade);
    assert_eq!(cocycles.len(), 1);
    assert_ne!(crate::topology::chain::pairing(&cocycles[0], &cycles[0]), 0);
  }

  /// On a closed manifold $diff K = nothing$, so the relative and absolute
  /// cohomologies coincide. The 2-sphere.
  #[test]
  fn closed_manifold_relative_equals_absolute() {
    let complex = two_sphere();
    for k in complex.dim().range_inclusive() {
      assert_eq!(
        complex.relative_cohomology_generators(k).len(),
        complex.cohomology_generators(k).len(),
        "grade {k}"
      );
    }
  }
}
