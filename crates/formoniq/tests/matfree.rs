//! The matrix-free [`ElementOperator`] and the assembled matrix are peers:
//! $A = sum_K P_K^top M_K P_K$ summed once or summed on every apply is the
//! same operator, so the two applies agree, square and rectangular alike.

use approx::assert_relative_eq;
use formoniq::galerkin::BilinearForm;
use formoniq::matfree::ElementOperator;
use formoniq::operators::WhitneyPairing;
use regge::lengths::mesh::MeshLengthsSq;
use simplicial::linalg::{CsrMatrix, Vector};
use simplicial::topology::complex::Complex;

fn mesh(dim: usize, refinement: usize) -> (Complex, MeshLengthsSq) {
  let coarse = Complex::unit(dim);
  let subdivision = coarse.refine(refinement);
  let geometry = MeshLengthsSq::unit(dim).refine(&subdivision, &coarse);
  (subdivision.into_complex(), geometry)
}

fn probe(len: usize) -> Vector {
  Vector::from_fn(len, |i, _| ((7 * i) % 13) as f64 - 6.0)
}

/// The matrix-free apply is the assembled matvec, on every operator and
/// grade.
///
/// The mixed forms are rectangular and pair two different grades, which is
/// where confusing the row and column incidences would show up and where a
/// square-only test would not. Checked on a refined mesh, so interior faces
/// carry contributions from several cells and the gather has something to
/// sum.
#[test]
fn the_matrix_free_apply_is_the_assembled_matvec() {
  /// One operator, both ways, on the same probe.
  fn agrees<E: BilinearForm>(topology: &Complex, geometry: &MeshLengthsSq, form: impl Fn() -> E) {
    let assembled: CsrMatrix = form().assemble(topology, geometry);
    let op = ElementOperator::new(topology, geometry, form());
    let x = probe(op.ncols());
    assert_relative_eq!(op.apply(&x), assembled * &x, epsilon = 1e-9);
  }

  for dim in 1..=3 {
    let (topology, geometry) = mesh(dim, 2);
    for grade in 0..=dim {
      agrees(&topology, &geometry, || WhitneyPairing::mass(dim, grade));
      if grade >= 1 {
        // Rectangular: a confusion of the row and column incidences would
        // pass unnoticed on the square forms alone.
        agrees(&topology, &geometry, || {
          WhitneyPairing::dif_trial(dim, grade)
        });
        agrees(&topology, &geometry, || {
          WhitneyPairing::dif_test(dim, grade)
        });
      }
      if grade < dim {
        agrees(&topology, &geometry, || {
          WhitneyPairing::dif_both(dim, grade + 1)
        });
      }
    }
  }
}
