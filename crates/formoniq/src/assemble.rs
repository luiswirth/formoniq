use crate::operators::{ElMatProvider, ElVecProvider};

use itertools::Itertools;
use simplicial::{
  geometry::metric::mesh::MeshLengthsSq,
  linalg::{CooMatrix, Vector},
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
      let elmat = elmat.eval(&metric);

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

#[cfg(test)]
mod test {
  use super::*;
  use crate::operators::HodgeMassElmat;

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
    let dim = 3;
    let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
    let lengths = coords.to_edge_lengths_sq(&topology);
    let round_trip = CellGramians::from_lengths(&topology, &lengths).to_edge_lengths_sq(&topology);

    for grade in 0..=dim {
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

    for dim in 1..=3 {
      let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
      let mut matrix = coords.into_matrix();
      matrix.row_mut(0).scale_mut(0.7);
      let spacetime = MeshCoords::with_ambient(matrix, gramian::Gramian::minkowski(dim));

      let from_coords = spacetime.to_edge_lengths_sq(&topology);
      let from_gramians = spacetime
        .to_cell_gramians(&topology)
        .to_edge_lengths_sq(&topology);

      for grade in 0..=dim {
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
