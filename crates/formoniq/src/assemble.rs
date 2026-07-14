use crate::operators::{ElMatProvider, ElVecProvider};

use common::linalg::nalgebra::{CooMatrix, Vector};
use itertools::Itertools;
use manifold::{geometry::metric::Geometry, topology::complex::Complex};

use rayon::prelude::*;

pub type GalMat = CooMatrix;
/// Assembly algorithm for the Galerkin Matrix.
pub fn assemble_galmat(
  topology: &Complex,
  geometry: &(impl Geometry + Sync),
  elmat: impl ElMatProvider,
) -> GalMat {
  let row_grade = elmat.row_grade();
  let col_grade = elmat.col_grade();

  let nsimps_row = topology.skeleton(row_grade).len();
  let nsimps_col = topology.skeleton(col_grade).len();

  let cells = topology.cells();
  let triplets: Vec<(usize, usize, f64)> = cells
    .handle_par_iter()
    .flat_map(|cell| {
      let metric = geometry.cell_metric(cell);
      let elmat = elmat.eval(&metric);

      let row_subs: Vec<_> = cell.faces(row_grade).collect();
      let col_subs: Vec<_> = cell.faces(col_grade).collect();

      let mut local_triplets = Vec::new();
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
  geometry: &(impl Geometry + Sync),
  elvec: impl ElVecProvider,
) -> GalVec {
  let grade = elvec.grade();
  let nsimps = topology.skeleton(grade).len();

  let cells = topology.cells();
  let entries: Vec<(usize, f64)> = cells
    .handle_par_iter()
    .flat_map(|cell| {
      let metric = geometry.cell_metric(cell);
      let elvec = elvec.eval(&metric, cell);

      let subs: Vec<_> = cell.faces(grade).collect();

      let mut local_entries = Vec::new();
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

  use common::linalg::nalgebra::Matrix;
  use manifold::{gen::cartesian::CartesianMeshInfo, geometry::metric::CellGramians};

  /// Cell Gramians are a first-class geometry: assembling against the per-cell
  /// metric tensors gives exactly the same Galerkin matrix as the edge-length
  /// geometry they were sampled from.
  #[test]
  fn cell_gramians_assemble_identically() {
    let dim = 3;
    let (topology, coords) = CartesianMeshInfo::new_unit(dim, 2).compute_coord_complex();
    let lengths = coords.to_edge_lengths(&topology);
    let gramians = CellGramians::from_geometry(&topology, &lengths);

    for grade in 0..=dim {
      let from_lengths = Matrix::from(&assemble_galmat(
        &topology,
        &lengths,
        HodgeMassElmat::new(dim, grade),
      ));
      let from_gramians = Matrix::from(&assemble_galmat(
        &topology,
        &gramians,
        HodgeMassElmat::new(dim, grade),
      ));
      approx::assert_relative_eq!(from_lengths, from_gramians, epsilon = 1e-12);
    }
  }
}
