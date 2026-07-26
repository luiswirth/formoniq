//! Element matrices as one constant tensor contracted with the cell geometry.
//!
//! A [`GramianLinearElMat`] evaluates as $M(g) = vol(K) L(Lambda^k (g^(-1)))$
//! with $L$ linear and independent of the cell. Writing $L$ out as a matrix in
//! the flattened bases turns *every* such element matrix into the same
//! expression,
//!
//! $
//!   "vec"(M_c) = vol_c dot A "vec"(G_c),
//! $
//!
//! with $A$ constant per $(n, k)$ and $G_c = Lambda^k (g_c^(-1))$ the cell's
//! multiform Gramian. Stacking the cells as columns makes the entire mesh's
//! element matrices one matrix product, which is what the batched and
//! matrix-free paths are: the same identity read with the cells stacked rather
//! than visited.

use super::{ElMat, ElMatProvider, GramianLinearElMat};

use exterior::{Dim, ExteriorGrade, exterior_dim, multiform_gramian};
use gramian::Metric;
use simplicial::{geometry::cell_volume, linalg::Matrix};

/// The constant tensor $A$ of a [`GramianLinearElMat`], in the flattened
/// bases: the whole cell-independent content of an element matrix.
///
/// The columns are indexed by the $P^2$ entries of the multiform Gramian
/// $Lambda^k (g^(-1))$ and the rows by the entries of the element matrix, both
/// flattened in column-major order. Applying it is one matrix product, so the
/// combinatorial sums an element matrix is defined by are paid once at
/// construction rather than once per cell.
#[derive(Debug, Clone)]
pub struct ElMatKernel {
  dim: Dim,
  form_grade: ExteriorGrade,
  nrows: usize,
  ncols: usize,
  coeff: Matrix,
}

impl ElMatKernel {
  /// Extract the constant tensor by probing the operator on a basis of
  /// Gramians.
  ///
  /// Linearity is what makes this an *identity* rather than an approximation:
  /// the $P^2$ probes determine $L$ completely. The probe matrices are the
  /// matrix units $E_(p q)$, which are not themselves Gramians of any metric;
  /// that is exactly the point, since the metrics do not span the space $L$ is
  /// linear on.
  pub fn new(op: &impl GramianLinearElMat) -> Self {
    let (dim, form_grade) = (op.dim(), op.form_grade());
    let p = exterior_dim(dim, form_grade);

    let mut probe = Matrix::zeros(p, p);
    let mut columns: Vec<Matrix> = Vec::with_capacity(p * p);
    for index in 0..p * p {
      probe[index] = 1.0;
      columns.push(op.eval_linear(&probe));
      probe[index] = 0.0;
    }

    let (nrows, ncols) = (columns[0].nrows(), columns[0].ncols());
    let coeff = Matrix::from_columns(
      &columns
        .iter()
        .map(|column| na::DVector::from_column_slice(column.as_slice()))
        .collect::<Vec<_>>(),
    );

    Self {
      dim,
      form_grade,
      nrows,
      ncols,
      coeff,
    }
  }

  /// The number of entries of the multiform Gramian a cell contributes,
  /// $P^2$ with $P = binom(n, k)$: the length of a [`Self::cell_column`].
  pub fn gramian_len(&self) -> usize {
    let p = exterior_dim(self.dim, self.form_grade);
    p * p
  }

  /// The element matrix shape, (rows, columns).
  pub fn shape(&self) -> (usize, usize) {
    (self.nrows, self.ncols)
  }

  /// The constant tensor itself, $A$ of shape $("nrows" dot "ncols") times P^2$.
  ///
  /// The device-side kernels take exactly this, and nothing else about the
  /// operator: it *is* the operator, once the cells are factored out.
  pub fn coeff(&self) -> &Matrix {
    &self.coeff
  }

  /// The per-cell geometry the kernel consumes: the flattened multiform
  /// Gramian scaled by the cell volume, $vol_c "vec"(G_c)$.
  ///
  /// Folding the volume in here rather than scaling afterwards is what leaves
  /// the contraction a plain matrix product with no per-column scaling.
  pub fn cell_column(&self, metric: &Metric) -> na::DVector<f64> {
    let gramian = multiform_gramian(metric, self.form_grade);
    cell_volume(metric) * na::DVector::from_column_slice(gramian.matrix().as_slice())
  }

  /// Evaluate a single element matrix, $vol_c dot A "vec"(G_c)$ reshaped.
  ///
  /// Agrees with the originating [`ElMatProvider::eval`] exactly, which is the
  /// law the kernel is tested against.
  pub fn eval(&self, metric: &Metric) -> ElMat {
    let flat = &self.coeff * self.cell_column(metric);
    Matrix::from_column_slice(self.nrows, self.ncols, flat.as_slice())
  }

  /// Evaluate every cell at once: the element matrices of the whole mesh as
  /// one matrix product $A X$, each column a flattened element matrix.
  ///
  /// `gramian_batch` holds the cells as columns, as [`Self::cell_column`]
  /// builds them. This is the entire arithmetic of assembly, in a single GEMM
  /// of shape $("nrows" dot "ncols") times P^2 times "ncells"$.
  pub fn eval_batch(&self, gramian_batch: &Matrix) -> Matrix {
    &self.coeff * gramian_batch
  }
}

/// The kernel of an operator, with the operator's own grades carried along.
///
/// [`ElMatKernel`] is deliberately ignorant of which spaces its rows and
/// columns index, since the contraction does not depend on it; assembly does,
/// so it is paired back up here.
#[derive(Debug, Clone)]
pub struct KernelElMat {
  kernel: ElMatKernel,
  row_grade: ExteriorGrade,
  col_grade: ExteriorGrade,
}

impl KernelElMat {
  pub fn new(op: &impl GramianLinearElMat) -> Self {
    Self {
      kernel: ElMatKernel::new(op),
      row_grade: op.row_grade(),
      col_grade: op.col_grade(),
    }
  }

  pub fn kernel(&self) -> &ElMatKernel {
    &self.kernel
  }
}

impl ElMatProvider for KernelElMat {
  fn row_grade(&self) -> ExteriorGrade {
    self.row_grade
  }
  fn col_grade(&self) -> ExteriorGrade {
    self.col_grade
  }
  fn eval(&self, metric: &Metric) -> ElMat {
    self.kernel.eval(metric)
  }
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::operators::{
    CodifDifElmat, CodifElmat, DifElmat, HodgeMassElmat, ScalarLumpedMassElmat,
  };

  use approx::assert_relative_eq;

  /// A metric that is neither the identity nor a multiple of it, so that an
  /// error vanishing on the reference cell cannot hide, and an indefinite one,
  /// since the factorization is a statement about linearity and owes nothing
  /// to the signature.
  fn metrics(dim: Dim) -> Vec<Metric> {
    let mut skewed = Matrix::identity(dim.index(), dim.index());
    for i in 0..dim.index() {
      skewed[(i, i)] = 1.0 + 0.5 * i as f64;
      if i + 1 < dim.index() {
        skewed[(i, i + 1)] = 0.25;
      }
    }
    let mut metrics = vec![
      Metric::standard(dim.index()),
      Metric::standard(dim.index()).pullback(&skewed),
    ];
    // A Minkowski metric needs a time axis, so the 0-dimensional cell has none.
    if dim.index() > 0 {
      metrics.push(Metric::minkowski(dim.index()));
    }
    metrics
  }

  /// The factorization is an identity, not an approximation: contracting the
  /// constant tensor with a cell's Gramian reproduces the operator's own
  /// evaluation, on every operator, dimension and grade.
  ///
  /// Swept on a non-reference geometry too, since the reference metric is the
  /// identity and would hide any error that vanishes there.
  #[test]
  fn kernel_reproduces_every_provider() {
    for dim in (0..=4).map(Dim::from) {
      for metric in metrics(dim) {
        check(&ScalarLumpedMassElmat::new(dim), &metric);
        for grade in dim.range_inclusive() {
          check(&HodgeMassElmat::new(dim, grade), &metric);
        }
        for grade in dim.range_inclusive().skip(1) {
          check(&DifElmat::new(dim, grade), &metric);
          check(&CodifElmat::new(dim, grade), &metric);
        }
        for grade in dim.range() {
          check(&CodifDifElmat::new(dim, grade), &metric);
        }
      }
    }
  }

  fn check(op: &impl GramianLinearElMat, metric: &Metric) {
    assert_relative_eq!(
      &ElMatKernel::new(op).eval(metric),
      &op.eval(metric),
      epsilon = 1e-12
    );
  }

  /// The batched path is the same arithmetic with the cells stacked: it must
  /// agree column by column with the per-cell one.
  #[test]
  fn batch_agrees_with_single() {
    let dim = Dim::new(3);
    let kernel = ElMatKernel::new(&HodgeMassElmat::new(dim, 1));
    let metrics = metrics(dim);

    let batch = Matrix::from_columns(
      &metrics
        .iter()
        .map(|metric| kernel.cell_column(metric))
        .collect::<Vec<_>>(),
    );
    let flat = kernel.eval_batch(&batch);

    for (icell, metric) in metrics.iter().enumerate() {
      let (nrows, ncols) = kernel.shape();
      let column = Matrix::from_column_slice(nrows, ncols, flat.column(icell).as_slice());
      assert_relative_eq!(&column, &kernel.eval(metric), epsilon = 1e-12);
    }
  }
}
