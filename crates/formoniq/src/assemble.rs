use crate::operators::{ElMatProvider, ElVecProvider};

use regge::lengths::mesh::MeshLengthsSq;
use simplicial::{
  linalg::{CooMatrix, CsrMatrix, Vector},
  topology::complex::Complex,
};

use rayon::prelude::*;

/// A coordinate-format matrix under construction: the three parallel arrays
/// [`CooMatrix`] is built from, accumulated rather than transposed into.
///
/// Only a scatter needs this. It exists so an assembly can hand back the arrays
/// in the shape the format already asks for, and so a parallel run can
/// accumulate one set per thread and concatenate in cell order, which keeps the
/// summation order of duplicate entries independent of how the work was split.
#[derive(Default)]
struct Triplets {
  rows: Vec<usize>,
  cols: Vec<usize>,
  values: Vec<f64>,
}
impl Triplets {
  fn reserve(&mut self, additional: usize) {
    self.rows.reserve(additional);
    self.cols.reserve(additional);
    self.values.reserve(additional);
  }
  fn push(&mut self, row: usize, col: usize, value: f64) {
    self.rows.push(row);
    self.cols.push(col);
    self.values.push(value);
  }
  fn concat(mut self, other: Self) -> Self {
    self.rows.extend(other.rows);
    self.cols.extend(other.cols);
    self.values.extend(other.values);
    self
  }
  fn into_arrays(self) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    (self.rows, self.cols, self.values)
  }
}

/// An assembled Galerkin matrix: compressed, because that is what an operator
/// is asked for.
///
/// Coordinate format is how a scatter *builds* a matrix, not what anyone wants
/// back: every caller here applies, factors, adds or restricts, and each of
/// those is a compressed-format operation. Handing back the triplet container
/// would export the assembly's own intermediate and leave the same conversion
/// at every call site.
pub type GalMat = CsrMatrix;
/// Assembly algorithm for the Galerkin Matrix.
///
/// The local-to-global map is streamed per cell rather than taken from a
/// materialized [`FaceIncidence`](simplicial::topology::incidence::FaceIncidence).
/// A scatter needs only the forward reading, which a cell already enumerates,
/// where that type's reason to exist is holding the converse alongside it: the
/// gather of [`crate::matfree`] is what needs both. Building it here costs
/// about twice the assembly time on a 3D grid of 80k cells and buys nothing.
pub fn assemble_galmat(
  topology: &Complex,
  geometry: &MeshLengthsSq,
  elmat: impl ElMatProvider,
) -> GalMat {
  let row_grade = elmat.row_grade();
  let col_grade = elmat.col_grade();

  let nsimps_row = topology.skeleton(row_grade).len();
  let nsimps_col = topology.skeleton(col_grade).len();

  let cells = topology.cells();
  // `fold`/`reduce`, so each thread scatters straight into the three parallel
  // arrays the COO format is and the arrays are concatenated in cell order.
  // Collecting triplets instead would materialize a `Vec` of tuples and then
  // transpose it into those arrays, one pass of the whole matrix twice over,
  // for a layout that is decided before the first cell is visited.
  //
  // The parallelism stays at cell granularity. A cell's contribution is
  // $binom(n+1, k+1) binom(n+1, k'+1)$ entries, single digits at the grades and
  // dimensions in reach, so handing one back to rayon as a splittable job would
  // pay scheduler overhead to divide work that fits in cache.
  let (rows, cols, values) = cells
    .handle_par_iter()
    .fold(Triplets::default, |mut acc, cell| {
      let metric = geometry.cell_metric(cell);
      let elmat = elmat.eval(&metric, cell);

      let row_subs: Vec<_> = cell.faces(row_grade).collect();
      let col_subs: Vec<_> = cell.faces(col_grade).collect();

      acc.reserve(row_subs.len() * col_subs.len());
      for (ilocal, iglobal) in row_subs.iter().enumerate() {
        for (jlocal, jglobal) in col_subs.iter().enumerate() {
          let val = elmat[(ilocal, jlocal)];
          if val != 0.0 {
            acc.push(iglobal.kidx(), jglobal.kidx(), val);
          }
        }
      }

      acc
    })
    .reduce(Triplets::default, Triplets::concat)
    .into_arrays();

  let coo = CooMatrix::try_from_triplets(nsimps_row, nsimps_col, rows, cols, values).unwrap();
  // The one place the triplets are summed and compressed, rather than once per
  // caller. Duplicate entries at a shared face add here, which is the scatter.
  GalMat::from(&coo)
}

pub type GalVec = Vector;
/// Assembly algorithm for the Galerkin Vector.
pub fn assemble_galvec(
  topology: &Complex,
  geometry: &MeshLengthsSq,
  elvec: impl ElVecProvider,
) -> GalVec {
  let grade = elvec.grade();
  let nsimps = topology.skeleton(grade).len();

  let cells = topology.cells();
  let entries: Vec<(usize, f64)> = cells
    .handle_par_iter()
    .flat_map_iter(|cell| {
      let metric = geometry.cell_metric(cell);
      let elvec = elvec.eval(&metric, cell);

      let subs: Vec<_> = cell.faces(grade).collect();

      let mut local_entries = Vec::with_capacity(subs.len());
      for (ilocal, &iglobal) in subs.iter().enumerate() {
        if elvec[ilocal] != 0.0 {
          local_entries.push((iglobal.kidx(), elvec[ilocal]));
        }
      }

      local_entries
    })
    .collect();

  let mut galvec = Vector::zeros(nsimps);
  for (irow, val) in entries {
    galvec[irow] += val;
  }
  galvec
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::operators::HodgeMassElmat;
  use simplicial::Dim;

  use simplicial::linalg::Matrix;

  use regge::{lengths::CellGramians, mesher::cartesian::CartesianGrid};

  /// Assembly consumes the edge-length primitive, so representation
  /// independence is a property of the conversions into it: routing a
  /// geometry through per-cell metrics
  /// ([`CellGramians`]) and reading them back as edge lengths reproduces the
  /// original lengths exactly, hence assembles identically. The derivation
  /// chain $"lengths" -> "metric" -> "lengths"$ commutes.
  #[test]
  fn cell_gramians_round_trip_assembles_identically() {
    let dim = Dim::new(3);
    let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
    let lengths = coords.to_edge_lengths_sq(&topology);
    let round_trip = CellGramians::from_lengths(&topology, &lengths).to_edge_lengths_sq(&topology);

    for grade in dim.range_inclusive() {
      let from_lengths = Matrix::from(&assemble_galmat(
        &topology,
        &lengths,
        HodgeMassElmat::new(dim, grade),
      ));
      let from_round_trip = Matrix::from(&assemble_galmat(
        &topology,
        &round_trip,
        HodgeMassElmat::new(dim, grade),
      ));
      approx::assert_relative_eq!(from_lengths, from_round_trip, epsilon = 1e-12);
    }
  }

  /// Every geometry source reduces to the same edge-length primitive on a
  /// Lorentzian mesh too: a Minkowski embedding, and the per-cell metrics it
  /// induces read back as edge lengths, yield identical Regge data and hence
  /// identical Galerkin matrices. This is Regge calculus doing what it was
  /// invented for, a simplicial spacetime carried by edge data alone, no
  /// coordinates in the assembly path.
  #[test]
  fn lorentzian_sources_reduce_to_the_same_regge_data() {
    use regge::coord::mesh::MeshCoords;

    for dim in (1..=3).map(Dim::from) {
      let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
      let mut matrix = coords.into_matrix();
      matrix.row_mut(0).scale_mut(0.7);
      let spacetime = MeshCoords::with_ambient(matrix, metric::Metric::minkowski(dim.index()));

      let from_coords = spacetime.to_edge_lengths_sq(&topology);
      let from_gramians = spacetime
        .to_cell_gramians(&topology)
        .to_edge_lengths_sq(&topology);

      for grade in dim.range_inclusive() {
        let a = Matrix::from(&assemble_galmat(
          &topology,
          &from_coords,
          HodgeMassElmat::new(dim, grade),
        ));
        let b = Matrix::from(&assemble_galmat(
          &topology,
          &from_gramians,
          HodgeMassElmat::new(dim, grade),
        ));
        approx::assert_relative_eq!(a, b, epsilon = 1e-12);
      }
    }
  }
}
