//! Simplicial cohomology over $ZZ$: universal coefficients against the
//! Betti number, Kronecker duality with homology, and the relative theory.

mod common;

use common::{annulus, is_cocycle, test_complexes, two_sphere};
use simplicial::Dim;
use simplicial::linalg::exact::IntegerMatrix;
use simplicial::topology::cohomology::kronecker_matrix;

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
  assert_ne!(
    simplicial::topology::chain::pairing(&cocycles[0], &cycles[0]),
    0
  );
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
