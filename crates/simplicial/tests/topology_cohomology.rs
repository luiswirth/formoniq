//! Kronecker duality between simplicial homology and cohomology over $ZZ$.

mod common;

use common::test_complexes;
use simplicial::linalg::exact::IntegerMatrix;
use simplicial::topology::cohomology::kronecker_matrix;

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
