//! The operator applied without ever assembling it.
//!
//! An assembled operator is a sum over cells,
//!
//! $
//!   A = sum_K P_K^top M_K P_K,
//! $
//!
//! with $P_K$ the gather of a cell's degrees of freedom. Assembly performs that
//! sum once and stores the result. A matrix-free apply performs it on every
//! matvec instead, and stores no element matrix and no sparsity pattern.
//!
//! The trade is memory for arithmetic. Where the assembled operator holds a
//! coefficient per nonzero, this holds only the cell metrics and the incidence,
//! and rebuilds each $M_K$ from them. On a CPU that is the losing direction for
//! speed and the winning one for size: it is how a problem whose assembled
//! matrix does not fit still gets solved. What it costs in capability is that a
//! direct factorization and an eigensolve need entries, so those still want
//! [`assemble_matrix`](crate::galerkin::assemble_matrix).
//!
//! # Gather, not scatter
//!
//! Performing the sum by visiting cells and adding each contribution into the
//! global result is a scatter, and two cells sharing a face write the same
//! entry. Visiting degrees of freedom and pulling in the cells at each is a
//! gather, where every output element is the property of one task and the
//! race is absent rather than synchronized away.
//!
//! Which direction is available is a property of the traversal, not of the
//! mathematics, and both are readings of one
//! [`simplicial::topology::incidence::FaceIncidence`]. The apply
//! here takes the gather in two stages: each cell writes $M_K P_K x$ to its own
//! slot, disjoint by construction, and each degree of freedom then sums the
//! slots incident to it.

use crate::galerkin::BilinearForm;

use iterative::{Jacobi, LinearOperator};
use metric::Metric;
use regge::lengths::mesh::MeshLengthsSq;
use simplicial::{
  linalg::Vector,
  topology::{complex::Complex, incidence::FaceIncidence},
};

use rayon::prelude::*;

/// An operator that rebuilds its element matrices on every apply.
///
/// Holds the per-cell metrics and the incidence of both grades, and nothing
/// per nonzero. This is the same operator
/// [`assemble_matrix`](crate::galerkin::assemble_matrix) produces, which is a
/// law the tests state rather than a remark.
///
/// Rectangular in general, since a mixed form pairs two grades; it is a
/// [`LinearOperator`] exactly when the two agree.
pub struct ElementOperator<'a, E> {
  topology: &'a Complex,
  form: E,
  /// One per cell, in cell order: the whole geometry the apply reads.
  metrics: Vec<Metric>,
  rows: FaceIncidence,
  cols: FaceIncidence,
}

impl<'a, E: BilinearForm> ElementOperator<'a, E> {
  /// Walk the mesh once, here. Every later apply is arithmetic on what this
  /// produced.
  pub fn new(topology: &'a Complex, geometry: &MeshLengthsSq, form: E) -> Self {
    let metrics = topology
      .cells()
      .handle_iter()
      .map(|cell| geometry.cell_metric(cell))
      .collect();
    Self {
      rows: FaceIncidence::new(topology, form.test_grade()),
      cols: FaceIncidence::new(topology, form.trial_grade()),
      topology,
      form,
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
  /// own slot. The second is over degrees of freedom and sums the slots at
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
        let elmat = self.form.element(&self.metrics[icell], cell);
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
impl<E: BilinearForm> LinearOperator for ElementOperator<'_, E> {
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
pub fn diagonal<E: BilinearForm>(op: &ElementOperator<'_, E>) -> Vector {
  assert_eq!(op.nrows(), op.ncols(), "a diagonal needs a square operator");
  let cells = op.topology.cells();
  let nlocal = op.rows.nlocal();
  let diagonals: Vec<f64> = cells
    .handle_par_iter()
    .flat_map_iter(|cell| {
      let elmat = op.form.element(&op.metrics[cell.kidx()], cell);
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

/// The Jacobi approximate inverse of a matrix-free operator, $B = omega
/// D^(-1)$: the ordinary [`Jacobi`], handed the diagonal [`diagonal`] gathered
/// instead of one read off an assembled matrix. A preconditioner does not care
/// where its diagonal came from.
pub fn jacobi<E: BilinearForm>(op: &ElementOperator<'_, E>, omega: f64) -> Jacobi {
  Jacobi::from_diagonal(&diagonal(op), omega)
}
