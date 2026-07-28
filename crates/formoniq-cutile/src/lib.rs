#![doc = include_str!("../README.md")]

pub mod spec;

#[cfg(feature = "cuda")]
pub mod device;
#[cfg(feature = "cuda")]
pub mod kernels;

use formoniq::{
  matfree::ElementOperator, operators::GramianLinearElMat, operators::kernel::ElMatKernel,
};
use simplicial::{
  geometry::metric::mesh::MeshLengthsSq, linalg::Vector, topology::complex::Complex,
};

use spec::Shapes;

/// The transpose of the local-to-global map, in compressed form: for each
/// degree of freedom, where its contributions sit in the flat per-cell array.
///
/// A mesh datum and nothing else --- no geometry, no operator, no grade beyond
/// the one that picked the skeleton --- so it is built once and reused by every
/// operator over the same rows. Its existence is what turns the scatter that
/// assembly wants into the gather the ownership discipline accepts: see
/// [`spec::gather_sum`].
#[derive(Clone, Debug)]
pub struct DofSegments {
  offsets: Vec<u32>,
  indices: Vec<u32>,
}

impl DofSegments {
  /// Invert a cell-major local-to-global map over `ndofs` degrees of freedom.
  ///
  /// Counting sort: one pass to size each segment, one to fill it. The result
  /// is a permutation of the input positions, so the total length is exactly
  /// the length of `dofs`, whatever the valences are.
  pub fn new(dofs: &[u32], ndofs: usize) -> Self {
    let mut offsets = vec![0u32; ndofs + 1];
    for &dof in dofs {
      offsets[dof as usize + 1] += 1;
    }
    for idof in 0..ndofs {
      offsets[idof + 1] += offsets[idof];
    }

    let mut cursor = offsets.clone();
    let mut indices = vec![0u32; dofs.len()];
    for (position, &dof) in dofs.iter().enumerate() {
      let slot = &mut cursor[dof as usize];
      indices[*slot as usize] = u32::try_from(position).expect("position fits in u32");
      *slot += 1;
    }

    Self { offsets, indices }
  }

  pub fn offsets(&self) -> &[u32] {
    &self.offsets
  }

  pub fn indices(&self) -> &[u32] {
    &self.indices
  }

  /// The largest number of cells meeting at one degree of freedom: the width
  /// [`Self::padded`] pads to.
  pub fn max_valence(&self) -> usize {
    self
      .offsets
      .windows(2)
      .map(|w| (w[1] - w[0]) as usize)
      .max()
      .unwrap_or(0)
  }

  /// The mean number of contributions per degree of freedom.
  ///
  /// Against [`Self::max_valence`] this is the efficiency of the padded form:
  /// their ratio is the fraction of a padded gather that does useful work.
  pub fn mean_valence(&self) -> f64 {
    let ndofs = self.offsets.len() - 1;
    if ndofs == 0 {
      return 0.0;
    }
    self.indices.len() as f64 / ndofs as f64
  }

  /// The same map with every segment padded to [`Self::max_valence`], for the
  /// fixed-shape device gather ([`spec::gather_sum_padded`]).
  ///
  /// Padding entries point at `nlocals`, the index of the trailing zero the
  /// caller appends to the contributions, so the kernel needs no branch: the
  /// padding gathers a zero and adds it.
  pub fn padded(&self, nlocals: usize) -> PaddedSegments {
    let width = self.max_valence();
    let ndofs = self.offsets.len() - 1;
    let sentinel = u32::try_from(nlocals).expect("index fits in u32");

    let mut table = vec![sentinel; ndofs * width];
    for idof in 0..ndofs {
      let (begin, end) = (self.offsets[idof] as usize, self.offsets[idof + 1] as usize);
      table[idof * width..idof * width + (end - begin)].copy_from_slice(&self.indices[begin..end]);
    }

    PaddedSegments { table, width }
  }
}

/// [`DofSegments`] with a constant segment width: the shape a tile program can
/// be written against.
#[derive(Clone, Debug)]
pub struct PaddedSegments {
  table: Vec<u32>,
  width: usize,
}

impl PaddedSegments {
  /// `ndofs × width`, row-major.
  pub fn table(&self) -> &[u32] {
    &self.table
  }

  pub fn width(&self) -> usize {
    self.width
  }
}

/// A [`GramianLinearElMat`] operator over a mesh, in the flat row-major layout
/// the device kernels read.
///
/// Holds exactly what a kernel launch needs and nothing that only makes sense
/// on the host: the constant tensor, the per-cell geometry, the two index maps
/// and the inverted row map. The GPU path uploads these buffers once and then
/// never touches the mesh again.
///
/// The apply runs exactly the stages the device runs, so this type is
/// simultaneously the host reference and the definition of what gets uploaded.
pub struct CellOperator {
  shapes: Shapes,
  /// The constant tensor, `nelmat × ngramian` row-major.
  coeff: Vec<f64>,
  /// The per-cell geometry, `ncells × ngramian` row-major.
  gramians: Vec<f64>,
  /// `ncells × ncols` row-major.
  col_dofs: Vec<u32>,
  /// The inverted `ncells × nrows` row map.
  rows: DofSegments,
  /// The same map at constant width, which is the shape a tile program takes.
  padded: PaddedSegments,
  ndofs_row: usize,
  ndofs_col: usize,
}

impl CellOperator {
  /// Flatten a [`formoniq`] operator into the device layout.
  ///
  /// The per-cell Gramians arrive as the columns of a nalgebra matrix, whose
  /// column-major storage already *is* the row-major cell-leading layout the
  /// device wants, so that buffer is taken as it stands. The constant tensor is
  /// genuinely transposed.
  pub fn new(topology: &Complex, geometry: &MeshLengthsSq, op: &impl GramianLinearElMat) -> Self {
    Self::from_element_operator(&ElementOperator::new(topology, geometry, op))
  }

  /// Flatten an already-built [`ElementOperator`], which is where the
  /// mathematics of the factorization lives.
  pub fn from_element_operator(op: &ElementOperator) -> Self {
    let kernel: &ElMatKernel = op.kernel();
    let (nrows, ncols) = kernel.shape();
    let shapes = Shapes {
      ncells: op.ncells(),
      ngramian: kernel.gramian_len(),
      nrows,
      ncols,
    };

    let coeff = kernel.coeff().transpose().as_slice().to_vec();
    let gramians = op.gramians().as_slice().to_vec();
    let (row_dofs, col_dofs) = op.dof_maps();

    let ndofs_row = row_dofs.iter().map(|&d| d as usize + 1).max().unwrap_or(0);
    let ndofs_col = col_dofs.iter().map(|&d| d as usize + 1).max().unwrap_or(0);
    let rows = DofSegments::new(row_dofs, ndofs_row);
    let padded = rows.padded(shapes.ncells * nrows);

    Self {
      shapes,
      coeff,
      gramians,
      col_dofs: col_dofs.to_vec(),
      rows,
      padded,
      ndofs_row,
      ndofs_col,
    }
  }

  pub fn shapes(&self) -> Shapes {
    self.shapes
  }

  pub fn coeff(&self) -> &[f64] {
    &self.coeff
  }

  pub fn gramians(&self) -> &[f64] {
    &self.gramians
  }

  pub fn col_dofs(&self) -> &[u32] {
    &self.col_dofs
  }

  pub fn rows(&self) -> &DofSegments {
    &self.rows
  }

  /// The constant-width form the device gather reads.
  pub fn padded(&self) -> &PaddedSegments {
    &self.padded
  }

  pub fn ndofs_row(&self) -> usize {
    self.ndofs_row
  }

  pub fn ndofs_col(&self) -> usize {
    self.ndofs_col
  }

  /// Every element matrix of the mesh, as one matrix product: the assembly
  /// kernel's output, `ncells × nelmat` row-major.
  ///
  /// Materializing them is not what the solve does --- the whole point of the
  /// matrix-free path is that it need not --- but it is what an assembled
  /// operator needs, and it is the simplest thing to check a device against.
  pub fn elmats(&self) -> Vec<f64> {
    let mut elmats = vec![0.0; self.shapes.ncells * self.shapes.nelmat()];
    spec::elmat_batch(self.shapes, &self.coeff, &self.gramians, &mut elmats);
    elmats
  }

  /// The host reference apply, in exactly the four stages the device runs.
  ///
  /// Gather the cell's degrees of freedom, apply the element matrices, gather
  /// the contributions belonging to each degree of freedom, reduce them. Two of
  /// the four are the same [`spec::gather`], which is the crate's one irregular
  /// access; the other two are regular and race-free.
  ///
  /// Allocating the intermediates on every call is host-side sloppiness that
  /// the device path does not share, where the buffers are allocated once and
  /// reused. It stays here because this is the reference, and clarity is what
  /// a reference is for.
  pub fn apply_rect(&self, x: &[f64]) -> Vec<f64> {
    let shapes = self.shapes;

    let mut cellx = vec![0.0; shapes.ncells * shapes.ncols];
    spec::gather(x, &self.col_dofs, &mut cellx);

    // One trailing zero, which the padded table's padding entries point at.
    let mut locals = vec![0.0; shapes.ncells * shapes.nrows + 1];
    spec::cell_matvec(
      shapes,
      &self.coeff,
      &self.gramians,
      &cellx,
      &mut locals[..shapes.ncells * shapes.nrows],
    );

    let mut gathered = vec![0.0; self.ndofs_row * self.padded.width()];
    spec::gather(&locals, self.padded.table(), &mut gathered);

    let mut y = vec![0.0; self.ndofs_row];
    spec::segment_reduce(&gathered, self.padded.width(), &mut y);
    y
  }
}

/// The host reference is itself a usable operator, which is what lets the
/// two-stage decomposition be tested through a full Krylov solve rather than
/// only on a single apply.
impl iterative::LinearOperator for CellOperator {
  fn dim(&self) -> usize {
    debug_assert_eq!(self.ndofs_row, self.ndofs_col);
    self.ndofs_row
  }

  fn apply(&self, x: &Vector) -> Vector {
    Vector::from_vec(self.apply_rect(x.as_slice()))
  }
}

#[cfg(test)]
mod test {
  use super::*;

  use formoniq::{
    assemble::assemble_galmat,
    operators::{CodifDifElmat, CodifElmat, DifElmat, HodgeMassElmat},
  };
  use simplicial::{Dim, linalg::Matrix, mesher::cartesian::CartesianGrid};

  use approx::assert_relative_eq;

  fn mesh(dim: Dim, subdivisions: usize) -> (Complex, MeshLengthsSq) {
    let (topology, coords) = CartesianGrid::new_unit(dim, subdivisions).triangulate();
    let geometry = coords.to_edge_lengths_sq(&topology);
    (topology, geometry)
  }

  /// The flattening into the device layout preserves the operator: the two
  /// stages the kernels run compose back to the matrix-free apply, which is
  /// itself the assembled one.
  ///
  /// This is the load-bearing test of the crate. It says the *decomposition
  /// into kernels* is right, which is the part a GPU cannot help you check:
  /// once it holds, a device run can only disagree by a transcription error in
  /// one kernel, and each of those has its own reference to be diffed against.
  #[test]
  fn the_two_stages_compose_to_the_assembled_operator() {
    for dim in (1..=3).map(Dim::from) {
      let (topology, geometry) = mesh(dim, 2);

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

  fn check(topology: &Complex, geometry: &MeshLengthsSq, op: impl GramianLinearElMat) {
    let cellop = CellOperator::new(topology, geometry, &op);
    let assembled = simplicial::linalg::CsrMatrix::from(&assemble_galmat(topology, geometry, op));

    let x = Vector::from_fn(assembled.ncols(), |i, _| ((i * 7 + 3) % 13) as f64 - 6.0);
    let expected = &assembled * &x;
    let got = Vector::from_vec(cellop.apply_rect(x.as_slice()));

    assert_eq!(cellop.ndofs_row(), assembled.nrows());
    assert_relative_eq!(&got, &expected, epsilon = 1e-10);
  }

  /// The batched assembly kernel produces the element matrices the operator
  /// evaluates one at a time, in the documented flattening.
  #[test]
  fn elmat_batch_matches_the_per_cell_evaluation() {
    let dim = Dim::new(3);
    let (topology, geometry) = mesh(dim, 2);
    let op = HodgeMassElmat::new(dim, 1);

    let element = ElementOperator::new(&topology, &geometry, &op);
    let cellop = CellOperator::from_element_operator(&element);
    let shapes = cellop.shapes();
    let batched = cellop.elmats();

    let kernel = element.kernel();
    for (icell, cell) in topology.cells().handle_iter().enumerate() {
      let expected = kernel.eval(&geometry.cell_metric(cell));
      let flat = &batched[icell * shapes.nelmat()..(icell + 1) * shapes.nelmat()];
      let got = Matrix::from_column_slice(shapes.nrows, shapes.ncols, flat);
      assert_relative_eq!(&got, &expected, epsilon = 1e-12);
    }
  }

  /// The inverted map is a permutation of the positions it inverts: every
  /// contribution is gathered exactly once, which is what makes the gather a
  /// faithful transpose of the scatter rather than an approximation of it.
  #[test]
  fn dof_segments_partition_the_contributions() {
    let dim = Dim::new(3);
    let (topology, geometry) = mesh(dim, 2);
    let element = ElementOperator::new(&topology, &geometry, &HodgeMassElmat::new(dim, 1));
    let (row_dofs, _) = element.dof_maps();
    let ndofs = row_dofs.iter().map(|&d| d as usize + 1).max().unwrap();

    let segments = DofSegments::new(row_dofs, ndofs);
    assert_eq!(segments.indices().len(), row_dofs.len());

    let mut seen = vec![false; row_dofs.len()];
    for (idof, window) in segments.offsets().windows(2).enumerate() {
      for &position in &segments.indices()[window[0] as usize..window[1] as usize] {
        assert_eq!(row_dofs[position as usize] as usize, idof);
        assert!(!seen[position as usize], "a contribution gathered twice");
        seen[position as usize] = true;
      }
    }
    assert!(seen.into_iter().all(|s| s), "a contribution never gathered");
  }

  /// The padded gather computes the same sums as the compressed one: the
  /// padding contributes exactly zero, on every operator and grade.
  ///
  /// It is the same law twice because the two are different kernels, and the
  /// device runs whichever the mesh's valence spread justifies.
  #[test]
  fn the_padded_gather_agrees_with_the_compressed_one() {
    for dim in (1..=3).map(Dim::from) {
      let (topology, geometry) = mesh(dim, 2);
      for grade in dim.range_inclusive() {
        let element = ElementOperator::new(&topology, &geometry, &HodgeMassElmat::new(dim, grade));
        let cellop = CellOperator::from_element_operator(&element);
        let shapes = cellop.shapes();

        let x = vec![1.0; cellop.ndofs_col()];
        let mut cellx = vec![0.0; shapes.ncells * shapes.ncols];
        spec::gather(&x, cellop.col_dofs(), &mut cellx);

        let nlocals = shapes.ncells * shapes.nrows;
        let mut locals = vec![0.0; nlocals + 1];
        spec::cell_matvec(
          shapes,
          cellop.coeff(),
          cellop.gramians(),
          &cellx,
          &mut locals[..nlocals],
        );

        let mut compressed = vec![0.0; cellop.ndofs_row()];
        spec::gather_sum(
          &locals[..nlocals],
          cellop.rows().offsets(),
          cellop.rows().indices(),
          &mut compressed,
        );

        let padded_map = cellop.padded();
        let mut padded = vec![0.0; cellop.ndofs_row()];
        spec::gather_sum_padded(&locals, padded_map.table(), padded_map.width(), &mut padded);

        assert_relative_eq!(
          &Vector::from_vec(padded),
          &Vector::from_vec(compressed),
          epsilon = 1e-12
        );
      }
    }
  }

  /// Totality at the degenerate boundary: a map onto no degrees of freedom
  /// inverts to empty segments rather than underflowing.
  #[test]
  fn dof_segments_are_total_when_empty() {
    let segments = DofSegments::new(&[], 0);
    assert_eq!(segments.offsets(), &[0]);
    assert!(segments.indices().is_empty());
    assert_eq!(segments.max_valence(), 0);
  }

  /// A Krylov solve driven through the flattened operator reaches the assembled
  /// answer, so the layout survives composition and not only a single apply.
  #[test]
  fn cg_solves_through_the_flattened_operator() {
    use iterative::{Identity, StopCriterion, krylov::cg};

    let dim = Dim::new(3);
    let (topology, geometry) = mesh(dim, 3);
    let op = HodgeMassElmat::new(dim, 1);

    let cellop = CellOperator::new(&topology, &geometry, &op);
    let assembled = simplicial::linalg::CsrMatrix::from(&assemble_galmat(&topology, &geometry, op));

    let ndofs = assembled.nrows();
    let b = Vector::from_fn(ndofs, |i, _| ((i * 5 + 1) % 11) as f64 - 5.0);
    let stop = StopCriterion::rtol(1e-10);

    let (x_cell, _) = cg(&cellop, &Identity::new(ndofs), &b, stop);
    let (x_assembled, _) = cg(&assembled, &Identity::new(ndofs), &b, stop);
    assert_relative_eq!(&x_cell, &x_assembled, epsilon = 1e-8);
  }

  /// The staged reduction agrees with the flat one to rounding, and only to
  /// rounding: floating-point addition is not associative, so the device's
  /// two-stage sum is a different sum of the same terms.
  #[test]
  fn staged_reduction_agrees_to_rounding() {
    let n = 10_000usize;
    let x: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
    let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.5).cos()).collect();

    let block = 256;
    let mut partials = vec![0.0; n.div_ceil(block)];
    spec::dot_partials(&x, &y, block, &mut partials);

    let staged: f64 = partials.iter().sum();
    let flat = spec::dot(&x, &y);
    assert!((staged - flat).abs() <= 1e-10 * flat.abs().max(1.0));
  }
}

#[cfg(test)]
mod valence_report {
  use super::*;
  use formoniq::operators::HodgeMassElmat;
  use simplicial::{Dim, mesher::cartesian::CartesianGrid};

  /// Not an assertion: a printed measurement of how much the padded gather
  /// wastes, which is what decides between the two gather kernels.
  #[test]
  fn report() {
    for dim in (2..=3).map(Dim::from) {
      let (topology, coords) = CartesianGrid::new_unit(dim, 8).triangulate();
      let geometry = coords.to_edge_lengths_sq(&topology);
      for grade in dim.range_inclusive() {
        let op = ElementOperator::new(&topology, &geometry, &HodgeMassElmat::new(dim, grade));
        let cellop = CellOperator::from_element_operator(&op);
        let rows = cellop.rows();
        println!(
          "dim {} grade {}: ncells {} ndofs {} mean valence {:.2} max {} efficiency {:.0}%",
          dim.index(),
          grade.index(),
          cellop.shapes().ncells,
          cellop.ndofs_row(),
          rows.mean_valence(),
          rows.max_valence(),
          100.0 * rows.mean_valence() / rows.max_valence() as f64,
        );
      }
    }
  }
}
