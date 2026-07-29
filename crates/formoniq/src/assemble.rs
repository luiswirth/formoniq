use crate::operators::{ElMatProvider, ElVecProvider};

use derham::decomposition::CellDofs;
use gramian::Metric;
use itertools::Itertools;
use simplicial::{
  atlas::Chart,
  geometry::metric::mesh::MeshLengthsSq,
  linalg::{CooMatrix, CsrMatrix, Vector},
  topology::complex::Complex,
};

use rayon::prelude::*;

pub type GalMat = CooMatrix;
/// Assembly algorithm for the Galerkin Matrix.
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
  // `flat_map_iter`, not `flat_map`: the parallelism is over cells, and each
  // cell's triplets number $binom(n+1, k)^2$ -- single digits at the grades and
  // dimensions in reach. `flat_map` would hand every such handful back to rayon
  // as a splittable parallel job, paying scheduler overhead per cell to divide
  // work that fits in cache. Measured ~2x on a 64k-cell 3D grid at grade 0.
  let triplets: Vec<(usize, usize, f64)> = cells
    .handle_par_iter()
    .flat_map_iter(|cell| {
      let metric = geometry.cell_metric(cell);
      let elmat = elmat.eval(&metric, cell);

      let row_subs: Vec<_> = cell.faces(row_grade).collect();
      let col_subs: Vec<_> = cell.faces(col_grade).collect();

      let mut local_triplets = Vec::with_capacity(row_subs.len() * col_subs.len());
      for (ilocal, &iglobal) in row_subs.iter().enumerate() {
        for (jlocal, &jglobal) in col_subs.iter().enumerate() {
          let val = elmat[(ilocal, jlocal)];
          if val != 0.0 {
            local_triplets.push((iglobal.kidx(), jglobal.kidx(), val));
          }
        }
      }

      local_triplets
    })
    .collect();

  let (rows, cols, values) = triplets.into_iter().multiunzip();
  GalMat::try_from_triplets(nsimps_row, nsimps_col, rows, cols, values).unwrap()
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

/// Assemble a Galerkin matrix through an explicit local-to-global map.
///
/// The general form of [`assemble_galmat`], which reads a dof off a
/// $k$-simplex of the cell: the first-order case of a [`CellDofs`]. Every
/// polynomial degree assembles through this.
///
/// `elmat` is evaluated once per cell and scattered; it must return a matrix of
/// shape `row_dofs.ndofs_local()` by `col_dofs.ndofs_local()`.
pub fn assemble_galmat_dofs(
  topology: &Complex,
  geometry: &MeshLengthsSq,
  row_dofs: &CellDofs,
  col_dofs: &CellDofs,
  elmat: impl Fn(&Metric, Chart) -> simplicial::linalg::Matrix + Sync,
) -> GalMat {
  let cells = topology.cells();
  let triplets: Vec<(usize, usize, f64)> = cells
    .handle_par_iter()
    .flat_map_iter(|cell| {
      let metric = geometry.cell_metric(cell);
      let elmat = elmat(&metric, cell);
      let rows = row_dofs.cell(cell.kidx());
      let cols = col_dofs.cell(cell.kidx());

      let mut local = Vec::with_capacity(rows.len() * cols.len());
      for (ilocal, &iglobal) in rows.iter().enumerate() {
        for (jlocal, &jglobal) in cols.iter().enumerate() {
          let value = elmat[(ilocal, jlocal)];
          if value != 0.0 {
            local.push((iglobal, jglobal, value));
          }
        }
      }
      local
    })
    .collect();

  let (rows, cols, values) = triplets.into_iter().multiunzip();
  GalMat::try_from_triplets(row_dofs.ndofs(), col_dofs.ndofs(), rows, cols, values).unwrap()
}

/// Scatter a cell-independent local matrix into a global one by *averaging*
/// rather than accumulating.
///
/// The exterior derivative is the same matrix on every cell, so a pair of dofs
/// is seen by every cell containing both, each contributing the identical
/// entry, and summing would multiply the operator by that multiplicity.
/// Visiting one cell per column does not avoid it: a basis function's
/// derivative is supported on all the cells containing its dof.
///
/// The multiplicity is counted by scattering ones through the same loop, so the
/// two sparsity patterns are identical by construction.
pub fn scatter_local_operator(
  topology: &Complex,
  row_dofs: &CellDofs,
  col_dofs: &CellDofs,
  local: &simplicial::linalg::Matrix,
) -> CsrMatrix {
  let mut values = CooMatrix::new(row_dofs.ndofs(), col_dofs.ndofs());
  let mut counts = CooMatrix::new(row_dofs.ndofs(), col_dofs.ndofs());

  for kidx in 0..topology.nsimplices(topology.dim()) {
    let rows = row_dofs.cell(kidx);
    let cols = col_dofs.cell(kidx);
    for (ilocal, &iglobal) in rows.iter().enumerate() {
      for (jlocal, &jglobal) in cols.iter().enumerate() {
        values.push(iglobal, jglobal, local[(ilocal, jlocal)]);
        counts.push(iglobal, jglobal, 1.0);
      }
    }
  }

  let mut values = CsrMatrix::from(&values);
  let counts = CsrMatrix::from(&counts);
  for (value, count) in values.values_mut().iter_mut().zip(counts.values()) {
    *value /= count;
  }
  values
}

/// Assemble a Galerkin vector through an explicit local-to-global map: the
/// counterpart of [`assemble_galmat_dofs`].
pub fn assemble_galvec_dofs(
  topology: &Complex,
  geometry: &MeshLengthsSq,
  dofs: &CellDofs,
  elvec: impl Fn(&Metric, Chart) -> Vector + Sync,
) -> GalVec {
  let contributions: Vec<(usize, f64)> = topology
    .cells()
    .handle_par_iter()
    .flat_map_iter(|cell| {
      let metric = geometry.cell_metric(cell);
      let elvec = elvec(&metric, cell);
      dofs
        .cell(cell.kidx())
        .iter()
        .copied()
        .zip(elvec.iter().copied())
        .collect::<Vec<_>>()
    })
    .collect();

  let mut galvec = GalVec::zeros(dofs.ndofs());
  for (global, value) in contributions {
    galvec[global] += value;
  }
  galvec
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::operators::HodgeMassElmat;
  use simplicial::Dim;

  use simplicial::{
    geometry::metric::CellGramians, linalg::Matrix, mesher::cartesian::CartesianGrid,
  };

  /// Assembly consumes the edge-length primitive, so representation
  /// independence is a property of the conversions *into* it: routing a
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
  /// *Lorentzian* mesh too: a Minkowski embedding, and the per-cell metrics it
  /// induces read back as edge lengths, yield identical Regge data and hence
  /// identical Galerkin matrices. This is Regge calculus doing what it was
  /// invented for -- a simplicial spacetime carried by edge data alone, no
  /// coordinates in the assembly path.
  #[test]
  fn lorentzian_sources_reduce_to_the_same_regge_data() {
    use simplicial::geometry::coord::mesh::MeshCoords;

    for dim in (1..=3).map(Dim::from) {
      let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
      let mut matrix = coords.into_matrix();
      matrix.row_mut(0).scale_mut(0.7);
      let spacetime = MeshCoords::with_ambient(matrix, gramian::Gramian::minkowski(dim.index()));

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
