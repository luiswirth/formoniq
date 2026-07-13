use crate::operators::{ElMatProvider, ElVecProvider};

use common::linalg::nalgebra::{CooMatrix, Vector};
use itertools::Itertools;
use manifold::{geometry::metric::mesh::MeshLengths, topology::complex::Complex};

use rayon::prelude::*;

pub type GalMat = CooMatrix;
/// Assembly algorithm for the Galerkin Matrix.
pub fn assemble_galmat(
  topology: &Complex,
  geometry: &MeshLengths,
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
      let geo = geometry.simplex_lengths(cell);
      let elmat = elmat.eval(&geo);

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
  geometry: &MeshLengths,
  elvec: impl ElVecProvider,
) -> GalVec {
  let grade = elvec.grade();
  let nsimps = topology.skeleton(grade).len();

  let cells = topology.cells();
  let entries: Vec<(usize, f64)> = cells
    .handle_par_iter()
    .flat_map(|cell| {
      let geo = geometry.simplex_lengths(cell);
      let elvec = elvec.eval(&geo, cell.simplex());

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
