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
//! matvec instead, and never stores $M_K$ at all.
//!
//! The trade is memory traffic for arithmetic. A sparse matvec reads one
//! coefficient per nonzero, of which a grade-$k$ Whitney operator has tens per
//! row; the matrix-free apply reads the cell's geometry, which is
//! $binom(n, k)^2$ numbers *shared* by the whole element matrix, and recomputes
//! the rest. On hardware where arithmetic is abundant and bandwidth is not,
//! which is what a GPU is, that is the favorable direction, and it is why this
//! is the form the device path takes.
//!
//! What makes it cheap enough to be worth it here is
//! [`crate::operators::kernel::ElMatKernel`]: the element matrix is
//! one small dense product away from the cell's Gramian, so recomputing it costs
//! a matrix product rather than the combinatorial sums of its definition.

use crate::operators::{
  ElMatProvider, GramianLinearElMat,
  kernel::{ElMatKernel, KernelElMat},
};

use exterior::ExteriorGrade;
use simplicial::{
  geometry::metric::mesh::MeshLengthsSq,
  linalg::{Matrix, Vector},
  topology::complex::Complex,
};

use rayon::prelude::*;

/// An operator that recomputes its element matrices on every apply.
///
/// Holds the mesh's geometry in the form the kernel consumes, one column of
/// $vol_K "vec"(Lambda^k g_K^(-1))$ per cell, together with the local-to-global
/// maps. The element matrices themselves are never stored: the operator's
/// memory is $O(binom(n,k)^2)$ per cell rather than $O(binom(n+1,k+1)^2)$, and
/// the assembled sparsity pattern does not exist at all.
///
/// This is the same operator [`assemble_galmat`](crate::assemble::assemble_galmat)
/// produces, which is a law the tests state, not a remark.
pub struct ElementOperator {
  kernel: ElMatKernel,
  /// The per-cell geometry, cells as columns.
  gramians: Matrix,
  /// The global row index of each cell's local row degrees of freedom,
  /// cell-major.
  row_dofs: Vec<u32>,
  /// The global column index of each cell's local column degrees of freedom,
  /// cell-major.
  col_dofs: Vec<u32>,
  nrows_local: usize,
  ncols_local: usize,
  ndofs_row: usize,
  ndofs_col: usize,
}

impl ElementOperator {
  /// Precompute the geometry and the local-to-global maps of an operator over
  /// the whole mesh.
  ///
  /// The cell loop runs once, here; every later apply is arithmetic on what it
  /// produced.
  pub fn new(topology: &Complex, geometry: &MeshLengthsSq, op: &impl GramianLinearElMat) -> Self {
    let elmat = KernelElMat::new(op);
    let (row_grade, col_grade) = (elmat.row_grade(), elmat.col_grade());
    let kernel = elmat.kernel().clone();

    let ndofs_row = topology.skeleton(row_grade).len();
    let ndofs_col = topology.skeleton(col_grade).len();

    let cells = topology.cells();
    let ncells = cells.len();

    let columns: Vec<_> = cells
      .handle_par_iter()
      .map(|cell| kernel.cell_column(&geometry.cell_metric(cell)))
      .collect();
    let gramians = Matrix::from_columns(&columns);

    let dofs = |grade: ExteriorGrade| -> Vec<u32> {
      cells
        .handle_iter()
        .flat_map(|cell| {
          cell
            .faces(grade)
            .map(|face| u32::try_from(face.kidx()).expect("dof index fits in u32"))
            .collect::<Vec<_>>()
        })
        .collect()
    };
    let row_dofs = dofs(row_grade);
    let col_dofs = dofs(col_grade);

    let (nrows_local, ncols_local) = kernel.shape();
    debug_assert_eq!(row_dofs.len(), ncells * nrows_local);
    debug_assert_eq!(col_dofs.len(), ncells * ncols_local);

    Self {
      kernel,
      gramians,
      row_dofs,
      col_dofs,
      nrows_local,
      ncols_local,
      ndofs_row,
      ndofs_col,
    }
  }

  /// The number of cells the sum runs over.
  pub fn ncells(&self) -> usize {
    self.gramians.ncols()
  }

  /// The per-cell geometry, cells as columns: the operator's entire data
  /// besides its constant tensor and its index maps.
  pub fn gramians(&self) -> &Matrix {
    &self.gramians
  }

  /// The constant tensor shared by every cell.
  pub fn kernel(&self) -> &ElMatKernel {
    &self.kernel
  }

  /// The local-to-global maps, cell-major: rows then columns.
  pub fn dof_maps(&self) -> (&[u32], &[u32]) {
    (&self.row_dofs, &self.col_dofs)
  }

  /// The rectangular apply $y = A x$, from the column space to the row space.
  ///
  /// Kept separate from the [`LinearOperator`](iterative::LinearOperator) impl
  /// because the mixed operators are genuinely rectangular, and a Krylov method
  /// is not the only consumer.
  pub fn apply_rect(&self, x: &Vector) -> Vector {
    assert_eq!(x.len(), self.ndofs_col);
    let mut y = Vector::zeros(self.ndofs_row);

    let (nrows, ncols) = (self.nrows_local, self.ncols_local);
    let mut local_x = Vector::zeros(ncols);
    for icell in 0..self.ncells() {
      let row_dofs = &self.row_dofs[icell * nrows..(icell + 1) * nrows];
      let col_dofs = &self.col_dofs[icell * ncols..(icell + 1) * ncols];

      for (ilocal, &jglobal) in col_dofs.iter().enumerate() {
        local_x[ilocal] = x[jglobal as usize];
      }

      // The element matrix, rebuilt and discarded: one product with the cell's
      // Gramian column, never stored beyond this iteration.
      let flat = self.kernel.coeff() * self.gramians.column(icell);
      let elmat = Matrix::from_column_slice(nrows, ncols, flat.as_slice());
      let local_y = elmat * &local_x;

      for (ilocal, &iglobal) in row_dofs.iter().enumerate() {
        y[iglobal as usize] += local_y[ilocal];
      }
    }
    y
  }
}

/// The square case is what a Krylov method consumes, and it is the one whose
/// row and column spaces coincide.
impl iterative::LinearOperator for ElementOperator {
  type Space = Vector;

  fn dim(&self) -> usize {
    debug_assert_eq!(
      self.ndofs_row, self.ndofs_col,
      "a Krylov operator must be square"
    );
    self.ndofs_row
  }

  fn apply(&self, x: &Vector) -> Vector {
    self.apply_rect(x)
  }
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::{
    assemble::assemble_galmat,
    operators::{CodifDifElmat, CodifElmat, DifElmat, HodgeMassElmat},
  };

  use simplicial::{Dim, mesher::cartesian::CartesianGrid};

  use approx::assert_relative_eq;

  /// The matrix-free apply *is* the assembled operator, not an approximation of
  /// it: the sum over cells is the same sum, performed at a different time.
  ///
  /// Checked on the rectangular mixed operators too, where the row and column
  /// spaces differ and a confusion between the two maps would otherwise go
  /// unnoticed.
  #[test]
  fn matfree_apply_equals_the_assembled_matvec() {
    for dim in (1..=3).map(Dim::from) {
      let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
      let geometry = coords.to_edge_lengths_sq(&topology);

      for grade in dim.range_inclusive() {
        check(&topology, &geometry, HodgeMassElmat::new(dim, grade));
      }
      for grade in dim.range_inclusive().skip(1) {
        check(&topology, &geometry, DifElmat::new(dim, grade));
        check(&topology, &geometry, CodifElmat::new(dim, grade));
      }
      for grade in dim.range() {
        check(&topology, &geometry, CodifDifElmat::new(dim, grade));
      }
    }
  }

  /// The point of the whole construction: a Krylov solve driven entirely by
  /// the matrix-free operator reaches the same solution as one driven by the
  /// assembled matrix, because [`iterative::krylov::cg`] asks its operator for
  /// nothing but the apply the two agree on.
  #[test]
  fn cg_solves_the_same_system_matrix_free() {
    use iterative::{Identity, StopCriterion, krylov::cg};

    let dim = Dim::new(3);
    let (topology, coords) = CartesianGrid::new_unit(dim, 3).triangulate();
    let geometry = coords.to_edge_lengths_sq(&topology);
    let op = HodgeMassElmat::new(dim, 1);

    let matfree = ElementOperator::new(&topology, &geometry, &op);
    let assembled = simplicial::linalg::CsrMatrix::from(&assemble_galmat(&topology, &geometry, op));

    let ndofs = assembled.nrows();
    let b = Vector::from_fn(ndofs, |i, _| ((i * 5 + 1) % 11) as f64 - 5.0);
    let stop = StopCriterion::rtol(1e-10);

    let (x_free, report_free) = cg(&matfree, &Identity::<Vector>::new(ndofs), &b, stop);
    let (x_assembled, report_assembled) = cg(&assembled, &Identity::<Vector>::new(ndofs), &b, stop);

    assert!(report_free.converged && report_assembled.converged);
    assert_eq!(report_free.iters, report_assembled.iters);
    assert_relative_eq!(&x_free, &x_assembled, epsilon = 1e-8);
  }

  fn check(topology: &Complex, geometry: &MeshLengthsSq, op: impl GramianLinearElMat) {
    let matfree = ElementOperator::new(topology, geometry, &op);
    let assembled = simplicial::linalg::CsrMatrix::from(&assemble_galmat(topology, geometry, op));

    let ncols = assembled.ncols();
    // A vector with no symmetry of its own, so a permuted or transposed index
    // map cannot coincidentally agree.
    let x = Vector::from_fn(ncols, |i, _| ((i * 7 + 3) % 13) as f64 - 6.0);

    assert_relative_eq!(&(&assembled * &x), &matfree.apply_rect(&x), epsilon = 1e-10);
  }
}
