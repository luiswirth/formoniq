//! The operator applied without ever assembling it.
//!
//! An assembled operator is a sum over cells,
//!
//! $
//!   A = sum_K P_K^top M_K P_K,
//! $
//!
//! with $P_K$ the gather of a cell's degrees of freedom. Assembly performs that
//! sum once and stores the result; a matrix-free apply performs it on every
//! matvec instead, and stores no element matrix and no sparsity pattern.
//!
//! The trade is memory for arithmetic. Where the assembled operator holds a
//! coefficient per nonzero, this holds only the cell metrics and the incidence,
//! and rebuilds each $M_K$ from them. On a CPU that is the losing direction for
//! *speed* and the winning one for *size*: it is how a problem whose assembled
//! matrix does not fit still gets solved. What it costs in capability is that a
//! direct factorization and an eigensolve need entries, so those still want
//! [`assemble_galmat`](crate::assemble::assemble_galmat).
//!
//! # Gather, not scatter
//!
//! Performing the sum by visiting cells and adding each contribution into the
//! global result is a *scatter*, and two cells sharing a face write the same
//! entry. Visiting degrees of freedom and pulling in the cells at each is a
//! *gather*, where every output element is the property of one task and the
//! race is absent rather than synchronized away.
//!
//! Which direction is available is a property of the traversal, not of the
//! mathematics, and both are readings of one
//! [`simplicial::topology::incidence::FaceIncidence`]. The apply
//! here takes the gather in two stages: each cell writes $M_K P_K x$ to its own
//! slot, disjoint by construction, and each degree of freedom then sums the
//! slots incident to it.

use crate::operators::ElMatProvider;

#[cfg(test)]
use approx::assert_relative_eq;

use gramian::Metric;
use iterative::{ApproxInverse, LinearOperator};
use simplicial::{
  geometry::metric::mesh::MeshLengthsSq,
  linalg::Vector,
  topology::{complex::Complex, incidence::FaceIncidence},
};

use rayon::prelude::*;

/// An operator that rebuilds its element matrices on every apply.
///
/// Holds the per-cell metrics and the incidence of both grades, and nothing
/// per nonzero. This is the same operator
/// [`assemble_galmat`](crate::assemble::assemble_galmat) produces, which is a
/// law the tests state rather than a remark.
///
/// Rectangular in general, since a mixed form pairs two grades; it is a
/// [`LinearOperator`] exactly when the two agree.
pub struct ElementOperator<'a, E> {
  topology: &'a Complex,
  elmat: E,
  /// One per cell, in cell order: the whole geometry the apply reads.
  metrics: Vec<Metric>,
  rows: FaceIncidence,
  cols: FaceIncidence,
}

impl<'a, E: ElMatProvider> ElementOperator<'a, E> {
  /// Walk the mesh once, here; every later apply is arithmetic on what this
  /// produced.
  pub fn new(topology: &'a Complex, geometry: &MeshLengthsSq, elmat: E) -> Self {
    let metrics = topology
      .cells()
      .handle_iter()
      .map(|cell| geometry.cell_metric(cell))
      .collect();
    Self {
      rows: FaceIncidence::new(topology, elmat.row_grade()),
      cols: FaceIncidence::new(topology, elmat.col_grade()),
      topology,
      elmat,
      metrics,
    }
  }

  pub fn nrows(&self) -> usize {
    self.rows.nfaces()
  }
  pub fn ncols(&self) -> usize {
    self.cols.nfaces()
  }

  /// $y = sum_K P_K^top M_K P_K x$, by gather.
  ///
  /// The first stage is over cells and writes each $M_K P_K x$ to that cell's
  /// own slot; the second is over degrees of freedom and sums the slots at
  /// each. Both are data-parallel without a lock, which is the point of taking
  /// the incidence in its two readings rather than one.
  pub fn apply(&self, x: &Vector) -> Vector {
    assert_eq!(x.len(), self.ncols(), "operator and vector disagree");
    let (nrows_local, ncols_local) = (self.rows.nlocal(), self.cols.nlocal());

    let cells = self.topology.cells();
    let contributions: Vec<f64> = cells
      .handle_par_iter()
      .flat_map_iter(|cell| {
        let icell = cell.kidx();
        let elmat = self.elmat.eval(&self.metrics[icell], cell);
        let gathered = Vector::from_iterator(
          ncols_local,
          self.cols.cell_faces(icell).iter().map(|&idof| x[idof]),
        );
        let local: Vec<f64> = (elmat * gathered).iter().copied().collect();
        local
      })
      .collect();

    Vector::from_vec(
      (0..self.nrows())
        .into_par_iter()
        .map(|idof| {
          self
            .rows
            .face_cells(idof)
            .iter()
            .map(|place| contributions[place.cell * nrows_local + place.position])
            .sum()
        })
        .collect(),
    )
  }
}

/// Square exactly when the form pairs one grade with itself, which the
/// dimension asserts, as it does for the assembled matrix.
impl<E: ElMatProvider> LinearOperator for ElementOperator<'_, E> {
  type Space = Vector;
  fn dim(&self) -> usize {
    debug_assert_eq!(self.nrows(), self.ncols(), "operator must be square");
    self.nrows()
  }
  fn apply(&self, x: &Vector) -> Vector {
    ElementOperator::apply(self, x)
  }
}

/// The diagonal of the operator, gathered from the element matrices.
///
/// Reachable without assembling anything: the diagonal of a sum is the sum of
/// the diagonals, and a cell contributes to entry $i$ only at the local
/// position $i$ takes in it. So the preconditioner a Krylov method most often
/// wants survives the matrix-free path, even though the trait that reads
/// entries does not.
pub fn diagonal<E: ElMatProvider>(op: &ElementOperator<'_, E>) -> Vector {
  assert_eq!(op.nrows(), op.ncols(), "a diagonal needs a square operator");
  let cells = op.topology.cells();
  let nlocal = op.rows.nlocal();
  let diagonals: Vec<f64> = cells
    .handle_par_iter()
    .flat_map_iter(|cell| {
      let elmat = op.elmat.eval(&op.metrics[cell.kidx()], cell);
      (0..nlocal).map(move |i| elmat[(i, i)]).collect::<Vec<_>>()
    })
    .collect();

  Vector::from_vec(
    (0..op.nrows())
      .into_par_iter()
      .map(|idof| {
        op.rows
          .face_cells(idof)
          .iter()
          .map(|place| diagonals[place.cell * nlocal + place.position])
          .sum()
      })
      .collect(),
  )
}

/// The Jacobi approximate inverse of a matrix-free operator: $B = D^(-1)$,
/// with $D$ read by [`diagonal`] rather than from an assembled matrix.
#[derive(Clone, Debug)]
pub struct MatrixFreeJacobi {
  inv_diag: Vector,
}

impl MatrixFreeJacobi {
  /// Panics on a zero diagonal entry, exactly as the assembled Jacobi does: a
  /// positive-definite operator has none.
  pub fn new<E: ElMatProvider>(op: &ElementOperator<'_, E>) -> Self {
    let diag = diagonal(op);
    assert!(
      diag.iter().all(|&d| d != 0.0),
      "a zero diagonal entry has no inverse"
    );
    Self {
      inv_diag: diag.map(f64::recip),
    }
  }
}

impl ApproxInverse for MatrixFreeJacobi {
  type Space = Vector;
  fn dim(&self) -> usize {
    self.inv_diag.len()
  }
  fn apply(&self, r: &Vector) -> Vector {
    self.inv_diag.component_mul(r)
  }
}

impl iterative::SelfAdjoint for MatrixFreeJacobi {}

#[cfg(test)]
mod test {
  use super::*;
  use crate::{
    assemble::assemble_galmat,
    operators::{CodifDifElmat, CodifElmat, DifElmat, HodgeMassElmat},
  };

  use iterative::{Identity, StopCriterion, krylov::cg};
  use simplicial::{
    geometry::metric::mesh::MeshLengthsSq, linalg::CsrMatrix, topology::complex::Complex,
  };

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
    fn agrees<E: ElMatProvider>(
      topology: &Complex,
      geometry: &MeshLengthsSq,
      elmat: impl Fn() -> E,
    ) {
      let assembled: CsrMatrix = (&assemble_galmat(topology, geometry, elmat())).into();
      let op = ElementOperator::new(topology, geometry, elmat());
      let x = probe(op.ncols());
      assert_relative_eq!(op.apply(&x), assembled * &x, epsilon = 1e-9);
    }

    for dim in 1..=3 {
      let (topology, geometry) = mesh(dim, 2);
      for grade in 0..=dim {
        agrees(&topology, &geometry, || HodgeMassElmat::new(dim, grade));
        if grade >= 1 {
          // Rectangular: a confusion of the row and column incidences would
          // pass unnoticed on the square forms alone.
          agrees(&topology, &geometry, || DifElmat::new(dim, grade));
          agrees(&topology, &geometry, || CodifElmat::new(dim, grade));
        }
        if grade < dim {
          agrees(&topology, &geometry, || CodifDifElmat::new(dim, grade));
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
    let dif = ElementOperator::new(&topology, &geometry, DifElmat::new(2, 1));
    let codif = ElementOperator::new(&topology, &geometry, CodifElmat::new(2, 1));
    assert_ne!(dif.nrows(), dif.ncols());
    assert_ne!(codif.nrows(), codif.ncols());
  }

  /// The gathered diagonal is the assembled diagonal, so the one preconditioner
  /// a matrix-free operator can still build is the right one.
  #[test]
  fn the_gathered_diagonal_is_the_assembled_one() {
    for dim in 1..=3 {
      let (topology, geometry) = mesh(dim, 2);
      for grade in 0..=dim {
        let assembled: CsrMatrix =
          (&assemble_galmat(&topology, &geometry, HodgeMassElmat::new(dim, grade))).into();
        let op = ElementOperator::new(&topology, &geometry, HodgeMassElmat::new(dim, grade));
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
        let assembled: CsrMatrix =
          (&assemble_galmat(&topology, &geometry, HodgeMassElmat::new(dim, grade))).into();
        let op = ElementOperator::new(&topology, &geometry, HodgeMassElmat::new(dim, grade));
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
    let op = ElementOperator::new(&topology, &geometry, CodifDifElmat::new(3, 0));
    let b = probe(op.nrows());
    let stop = StopCriterion::rtol(1e-10);

    let (_, plain) = cg(&op, &Identity::new(b.len()), &b, stop);
    let (_, jacobi) = cg(&op, &MatrixFreeJacobi::new(&op), &b, stop);
    assert!(
      jacobi.iters < plain.iters,
      "{} vs {}",
      jacobi.iters,
      plain.iters
    );
  }
}
