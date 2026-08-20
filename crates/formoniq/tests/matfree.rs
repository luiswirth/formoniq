//! The matrix-free [`ElementOperator`]: its apply is the assembled matvec
//! (square and rectangular alike), the gathered diagonal is the assembled
//! one, and driving CG through it reaches the same solution in the same
//! number of iterations as the assembled operator.

use approx::assert_relative_eq;
use formoniq::galerkin::BilinearForm;
use formoniq::matfree::{ElementOperator, diagonal, jacobi};
use formoniq::operators::WhitneyPairing;
use iterative::{Identity, StopCriterion, krylov::cg};
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

/// The mixed forms really are rectangular, so the test above is exercising
/// the two incidences against each other rather than one twice.
#[test]
fn the_mixed_forms_pair_two_grades() {
  // Refinement 2, not 1: an unrefined triangle has as many edges as vertices,
  // and the shapes would coincide for a reason that says nothing.
  let (topology, geometry) = mesh(2, 2);
  let dif = ElementOperator::new(&topology, &geometry, WhitneyPairing::dif_trial(2, 1));
  let mixed = ElementOperator::new(&topology, &geometry, WhitneyPairing::dif_test(2, 1));
  assert_ne!(dif.nrows(), dif.ncols());
  assert_ne!(mixed.nrows(), mixed.ncols());
}

/// The gathered diagonal is the assembled diagonal, so the one preconditioner
/// a matrix-free operator can still build is the right one.
#[test]
fn the_gathered_diagonal_is_the_assembled_one() {
  for dim in 1..=3 {
    let (topology, geometry) = mesh(dim, 2);
    for grade in 0..=dim {
      let assembled: CsrMatrix = WhitneyPairing::mass(dim, grade).assemble(&topology, &geometry);
      let op = ElementOperator::new(&topology, &geometry, WhitneyPairing::mass(dim, grade));
      let expected = Vector::from_fn(op.nrows(), |i, _| {
        assembled.get_entry(i, i).unwrap().into_value()
      });
      assert_relative_eq!(diagonal(&op), expected, epsilon = 1e-9);
    }
  }
}

/// Conjugate gradients driven matrix-free reaches the same solution in the
/// same number of iterations, since it asks for nothing but the apply.
///
/// The iteration count is the sharp part: CG is a deterministic recurrence
/// in the inner products, so an apply that differed anywhere would diverge
/// from the assembled run rather than merely land within tolerance.
#[test]
fn conjugate_gradients_does_not_notice_the_difference() {
  for dim in 1..=3 {
    let (topology, geometry) = mesh(dim, 2);
    for grade in 0..=dim {
      let assembled: CsrMatrix = WhitneyPairing::mass(dim, grade).assemble(&topology, &geometry);
      let op = ElementOperator::new(&topology, &geometry, WhitneyPairing::mass(dim, grade));
      let b = probe(op.nrows());
      let stop = StopCriterion::rtol(1e-10);

      let (x_assembled, ra) = cg(&assembled, &Identity::new(b.len()), &b, stop);
      let (x_free, rf) = cg(&op, &Identity::new(b.len()), &b, stop);
      assert_eq!(ra.iters, rf.iters, "dim {dim} grade {grade}");
      assert_relative_eq!(x_assembled, x_free, epsilon = 1e-9);
    }
  }
}

/// Preconditioning by the gathered diagonal cuts the iteration count, which
/// is what makes it worth having.
#[test]
fn the_matrix_free_jacobi_preconditions() {
  let (topology, geometry) = mesh(3, 2);
  let op = ElementOperator::new(&topology, &geometry, WhitneyPairing::dif_both(3, 1));
  let b = probe(op.nrows());
  let stop = StopCriterion::rtol(1e-10);

  let (_, plain) = cg(&op, &Identity::new(b.len()), &b, stop);
  let (_, jacobi_report) = cg(&op, &jacobi(&op, 1.0), &b, stop);
  assert!(
    jacobi_report.iters < plain.iters,
    "{} vs {}",
    jacobi_report.iters,
    plain.iters
  );
}
