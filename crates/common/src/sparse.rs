use crate::util::{CumsumExt, IterAllEqExt};

use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use std::{
  fs::File,
  io::{BufReader, BufWriter, Write},
  mem,
};

pub trait CooMatrixExt {
  fn neg(self) -> Self;
  fn block(block_grid: &[&[&Self]]) -> Self;
  fn set_zero<F>(&mut self, predicate: F)
  where
    F: Fn(usize, usize) -> bool;
  fn grow(&mut self, nrows_added: usize, ncols_added: usize);
  fn transpose(self) -> Self;
}

impl CooMatrixExt for nas::CooMatrix<f64> {
  fn grow(&mut self, nrows_added: usize, ncols_added: usize) {
    let nrows = self.nrows() + nrows_added;
    let ncols = self.ncols() + ncols_added;
    let (rows, cols, values) = mem::replace(self, Self::new(0, 0)).disassemble();
    *self = Self::try_from_triplets(nrows, ncols, rows, cols, values).unwrap()
  }

  fn transpose(self) -> Self {
    let nrows = self.nrows();
    let ncols = self.ncols();
    let (rows, cols, values) = self.disassemble();
    Self::try_from_triplets(ncols, nrows, cols, rows, values).unwrap()
  }

  fn neg(self) -> Self {
    let nrows = self.nrows();
    let ncols = self.ncols();
    let (rows, cols, mut values) = self.disassemble();
    for value in &mut values {
      *value = -*value;
    }
    Self::try_from_triplets(nrows, ncols, rows, cols, values).unwrap()
  }

  fn set_zero<F>(&mut self, predicate: F)
  where
    F: Fn(usize, usize) -> bool,
  {
    let nrows = self.nrows();
    let ncols = self.ncols();
    let (mut rows, mut cols, mut vals) = mem::replace(self, Self::new(0, 0)).disassemble();
    let mut i = 0;
    while i < rows.len() {
      let r = rows[i];
      let c = cols[i];
      if predicate(r, c) {
        rows.swap_remove(i);
        cols.swap_remove(i);
        vals.swap_remove(i);
      } else {
        i += 1;
      }
    }
    *self = Self::try_from_triplets(nrows, ncols, rows, cols, vals).unwrap()
  }

  /// Concatenates a matrix block grid row-wise and column-wise, automatically computing offsets.
  fn block(block_grid: &[&[&Self]]) -> Self {
    block_grid
      .iter()
      .map(|row| row.len())
      .all_eq()
      .expect("Each block row must contain the same number of matrices.");

    let mut row_offsets: Vec<usize> = block_grid
      .iter()
      .map(|row| {
        let nrows = row.first().map_or(0, |m| m.nrows());
        assert!(row.iter().all(|m| nrows == m.nrows()));
        nrows
      })
      .cumsum()
      .collect();
    let nrows_total = row_offsets.pop().unwrap_or(0);
    row_offsets.insert(0, 0);

    let mut col_offsets: Vec<usize> = block_grid
      .iter()
      .map(|row| row.iter().map(|mat| mat.ncols()).cumsum().collect())
      .all_eq()
      .expect("Each row must have matrices at the same offsets.");
    let ncols_total = col_offsets.pop().unwrap_or(0);
    col_offsets.insert(0, 0);

    let mut result = Self::zeros(nrows_total, ncols_total);

    for (i, row) in block_grid.iter().enumerate() {
      for (j, block) in row.iter().enumerate() {
        let row_offset = row_offsets[i];
        let col_offset = col_offsets[j];

        for (r, c, &v) in block.triplet_iter() {
          result.push(row_offset + r, col_offset + c, v);
        }
      }
    }

    result
  }
}

const PETSC_SOLVER_PATH: &str = "./petsc-solver";

const PETSC_MAT_FILE_CLASSID: i32 = 1211216;
const PETSC_VEC_FILE_CLASSID: i32 = 1211214;

pub fn petsc_write_matrix(matrix: &nas::CsrMatrix<f64>, filename: &str) -> std::io::Result<()> {
  let file = File::create(filename)?;
  let mut writer = BufWriter::new(file);

  writer.write_i32::<BigEndian>(PETSC_MAT_FILE_CLASSID)?;

  let nrows = matrix.nrows() as i32;
  let ncols = matrix.ncols() as i32;
  let nnz = matrix.nnz() as i32;
  writer.write_i32::<BigEndian>(nrows)?;
  writer.write_i32::<BigEndian>(ncols)?;
  writer.write_i32::<BigEndian>(nnz)?;

  let row_offsets = matrix.row_offsets();
  for i in 0..nrows as usize {
    let row_nnz = (row_offsets[i + 1] - row_offsets[i]) as i32;
    writer.write_i32::<BigEndian>(row_nnz)?;
  }

  let col_indices = matrix.col_indices();
  for &col in col_indices {
    writer.write_i32::<BigEndian>(col as i32)?;
  }

  let values = matrix.values();
  for &value in values {
    writer.write_f64::<BigEndian>(value)?;
  }

  writer.flush()?;
  Ok(())
}

pub fn petsc_write_vector(vector: &na::DVector<f64>, filename: &str) -> std::io::Result<()> {
  let file = File::create(filename)?;
  let mut writer = BufWriter::new(file);

  writer.write_i32::<BigEndian>(PETSC_VEC_FILE_CLASSID)?;

  let nrows = vector.nrows() as i32;
  writer.write_i32::<BigEndian>(nrows)?;

  for &value in vector {
    writer.write_f64::<BigEndian>(value)?;
  }

  writer.flush()?;
  Ok(())
}

pub fn petsc_read_vector(filename: &str) -> std::io::Result<na::DVector<f64>> {
  let file = File::open(filename)?;
  let mut reader = BufReader::new(file);

  let magic = reader.read_i32::<BigEndian>()?;
  assert_eq!(magic, PETSC_VEC_FILE_CLASSID);

  let nrows = reader.read_i32::<BigEndian>()? as usize;

  let mut vector = na::DVector::zeros(nrows);
  for i in 0..nrows {
    vector[i] = reader.read_f64::<BigEndian>()?;
  }

  Ok(vector)
}

pub fn petsc_read_eigenvals(filename: &str) -> std::io::Result<na::DVector<f64>> {
  let file = File::open(filename)?;
  let mut reader = BufReader::new(file);

  let neigenvals = reader.read_i32::<BigEndian>()? as usize;
  let mut eigenvals = na::DVector::zeros(neigenvals);

  for i in 0..neigenvals {
    eigenvals[i] = reader.read_f64::<BigEndian>()?;
  }

  Ok(eigenvals)
}

pub fn petsc_read_eigenvecs(filename: &str) -> std::io::Result<nalgebra::DMatrix<f64>> {
  let file = File::open(filename)?;
  let mut reader = BufReader::new(file);

  let nrows = reader.read_i32::<BigEndian>()? as usize;
  let ncols = reader.read_i32::<BigEndian>()? as usize;

  let mut data = Vec::with_capacity(nrows * ncols);
  for _ in 0..ncols {
    let magic = reader.read_i32::<BigEndian>()?;
    assert_eq!(magic, PETSC_VEC_FILE_CLASSID);

    let this_nrows = reader.read_i32::<BigEndian>()? as usize;
    assert_eq!(this_nrows, nrows);

    for _ in 0..nrows {
      data.push(reader.read_f64::<BigEndian>()?);
    }
  }
  Ok(na::DMatrix::from_column_slice(nrows, ncols, &data))
}

pub fn petsc_ghiep(
  lhs: &nas::CsrMatrix<f64>,
  rhs: &nas::CsrMatrix<f64>,
  neigen_values: usize,
) -> (na::DVector<f64>, na::DMatrix<f64>) {
  petsc_write_matrix(lhs, &format!("{PETSC_SOLVER_PATH}/in/A.bin")).unwrap();
  petsc_write_matrix(rhs, &format!("{PETSC_SOLVER_PATH}/in/B.bin")).unwrap();

  let binary = "./ghiep.out";
  #[rustfmt::skip]
  let args = [
    "-st_pc_factor_mat_solver_type", "mumps",
    "-st_type", "sinvert",
    "-st_shift", "0.1",
    "-eps_target", "0.",
    "-eps_nev", &neigen_values.to_string(),
  ];

  let status = std::process::Command::new(binary)
    .current_dir(PETSC_SOLVER_PATH)
    .args(args)
    .status()
    .unwrap();
  assert!(status.success());

  let eigenvals = petsc_read_eigenvals(&format!("{PETSC_SOLVER_PATH}/out/eigenvals.bin")).unwrap();
  let eigenvecs = petsc_read_eigenvecs(&format!("{PETSC_SOLVER_PATH}/out/eigenvecs.bin")).unwrap();

  (eigenvals, eigenvecs)
}

pub fn petsc_saddle_point(lhs: &nas::CsrMatrix<f64>, rhs: &na::DVector<f64>) -> na::DVector<f64> {
  petsc_write_matrix(lhs, &format!("{PETSC_SOLVER_PATH}/in/A.bin")).unwrap();
  petsc_write_vector(rhs, &format!("{PETSC_SOLVER_PATH}/in/b.bin")).unwrap();

  let binary = "./hils.out";
  #[rustfmt::skip]
  let args: [&str; 0] = [
    //"-ksp_type", "minres",
    //"-pc_type", "lu",
    //"-ksp_max_it", "1000",
    //"-ksp_rtol", "1e-9",
  ];

  let status = std::process::Command::new(binary)
    .current_dir(PETSC_SOLVER_PATH)
    .args(args)
    .status()
    .unwrap();
  assert!(status.success());

  petsc_read_vector(&format!("{PETSC_SOLVER_PATH}/out/x.bin")).unwrap()
}
