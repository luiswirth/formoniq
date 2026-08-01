//! The Galerkin discretization: a form, a finite-dimensional subspace, and the
//! matrix the first becomes on the second.
//!
//! A [`BilinearForm`] is the continuous object $b(v, u)$; Galerkin is what
//! happens when it is restricted to a subspace, $A_h = i^* b i$, so the word
//! names the passage and the matrix, never the form. The restriction is
//! metric-free and purely a change of what the arguments range over, which is
//! why the same form serves a cell, a mesh, and a mesh with boundary conditions
//! imposed.
//!
//! A form is evaluated at two scopes, and they are the same object read
//! twice: [`BilinearForm::element`] on one cell in its own chart, and
//! [`BilinearForm::assemble`] over the whole mesh. Assembly is the sum
//! $A = sum_K P_K^top A_K P_K$ of the first over all cells, with $P_K$ the
//! local-to-global map the cell's faces already enumerate.

use metric::Metric;
use multialgebra::ExteriorGrade;
use regge::lengths::mesh::MeshLengthsSq;
use simplicial::{
  atlas::Chart,
  linalg::{CooMatrix, CsrMatrix, Matrix, Vector},
  topology::complex::Complex,
};

use rayon::prelude::*;

/// A bilinear form $b(v, u)$ on the discrete de Rham complex, with $v$ the test
/// argument and $u$ the trial one.
///
/// The two arguments stand on the left and the right of the form, and the
/// matrix keeps them there: $A_(sigma tau) = b(v_sigma, u_tau)$, so the test
/// side indexes rows, the trial side columns, and the pairing is $v^top A u$
/// with nothing transposed anywhere. A form whose test side carries the
/// exterior derivative therefore has fewer rows than columns by one grade,
/// which is $angle.l u, dif tau angle.r$ read literally.
///
/// The two methods are one form at two scopes, local and global. An
/// implementor writes the local one, on a single cell in that cell's own
/// chart, and assembly over the mesh follows from it.
pub trait BilinearForm: Sync {
  /// The grade of the test argument, hence of the rows.
  fn test_grade(&self) -> ExteriorGrade;
  /// The grade of the trial argument, hence of the columns.
  fn trial_grade(&self) -> ExteriorGrade;

  /// The form on one cell, in that cell's chart: the element matrix.
  fn element(&self, metric: &Metric, chart: Chart) -> Matrix;

  /// The form on the whole mesh: the Galerkin matrix.
  fn assemble(&self, topology: &Complex, geometry: &MeshLengthsSq) -> GalerkinMatrix
  where
    Self: Sized,
  {
    assemble_matrix(topology, geometry, self)
  }
}

/// A linear form $ell(v)$ on the discrete de Rham complex: a [`BilinearForm`]
/// with one argument fewer, and the one it keeps is the test argument.
///
/// A functional, so its Galerkin realization is a vector of the values it takes
/// on the basis, $ell_sigma = ell(v_sigma)$, indexed exactly as the rows of a
/// bilinear form are.
pub trait LinearForm: Sync {
  /// The grade of the test argument, hence of the entries.
  fn test_grade(&self) -> ExteriorGrade;

  /// The form on one cell, in that cell's chart: the element vector.
  fn element(&self, metric: &Metric, chart: Chart) -> Vector;

  /// The form on the whole mesh: the Galerkin vector.
  fn assemble(&self, topology: &Complex, geometry: &MeshLengthsSq) -> GalerkinVector
  where
    Self: Sized,
  {
    assemble_vector(topology, geometry, self)
  }
}

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
pub type GalerkinMatrix = CsrMatrix;
/// Assembly algorithm for the Galerkin matrix, the sum
/// $A = sum_K P_K^top A_K P_K$ of a form's element matrices.
///
/// Reached as [`BilinearForm::assemble`]; a free function because it is the
/// algorithm rather than the form's own business.
///
/// The local-to-global map is streamed per cell rather than taken from a
/// materialized [`FaceIncidence`](simplicial::topology::incidence::FaceIncidence).
/// A scatter needs only the forward reading, which a cell already enumerates,
/// where that type's reason to exist is holding the converse alongside it: the
/// gather of [`crate::matfree`] is what needs both. Building it here costs
/// about twice the assembly time on a 3D grid of 80k cells and buys nothing.
pub fn assemble_matrix(
  topology: &Complex,
  geometry: &MeshLengthsSq,
  form: &impl BilinearForm,
) -> GalerkinMatrix {
  let test_grade = form.test_grade();
  let trial_grade = form.trial_grade();

  let nsimps_test = topology.skeleton(test_grade).len();
  let nsimps_trial = topology.skeleton(trial_grade).len();

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
      let elmat = form.element(&metric, cell);

      let test_subs: Vec<_> = cell.faces(test_grade).collect();
      let trial_subs: Vec<_> = cell.faces(trial_grade).collect();

      acc.reserve(test_subs.len() * trial_subs.len());
      for (ilocal, iglobal) in test_subs.iter().enumerate() {
        for (jlocal, jglobal) in trial_subs.iter().enumerate() {
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

  let coo = CooMatrix::try_from_triplets(nsimps_test, nsimps_trial, rows, cols, values).unwrap();
  // The one place the triplets are summed and compressed, rather than once per
  // caller. Duplicate entries at a shared face add here, which is the scatter.
  GalerkinMatrix::from(&coo)
}

pub type GalerkinVector = Vector;
/// Assembly algorithm for the Galerkin vector, reached as
/// [`LinearForm::assemble`].
pub fn assemble_vector(
  topology: &Complex,
  geometry: &MeshLengthsSq,
  form: &impl LinearForm,
) -> GalerkinVector {
  let grade = form.test_grade();
  let nsimps = topology.skeleton(grade).len();

  let cells = topology.cells();
  let entries: Vec<(usize, f64)> = cells
    .handle_par_iter()
    .flat_map_iter(|cell| {
      let metric = geometry.cell_metric(cell);
      let elvec = form.element(&metric, cell);

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
  use crate::operators::WhitneyPairing;
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
      let from_lengths =
        Matrix::from(&WhitneyPairing::mass(dim, grade).assemble(&topology, &lengths));
      let from_round_trip =
        Matrix::from(&WhitneyPairing::mass(dim, grade).assemble(&topology, &round_trip));
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
        let a = Matrix::from(&WhitneyPairing::mass(dim, grade).assemble(&topology, &from_coords));
        let b = Matrix::from(&WhitneyPairing::mass(dim, grade).assemble(&topology, &from_gramians));
        approx::assert_relative_eq!(a, b, epsilon = 1e-12);
      }
    }
  }
}
