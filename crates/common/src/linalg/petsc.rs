use std::{
  fs::File,
  io::{BufReader, BufWriter, Write},
};

use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};

use super::nalgebra::{CsrMatrix, Matrix, Vector};

const PETSC_SOLVER_PATH: &str = "./petsc-solver";

const PETSC_MAT_FILE_CLASSID: i32 = 1211216;
const PETSC_VEC_FILE_CLASSID: i32 = 1211214;

pub fn petsc_write_matrix(matrix: &CsrMatrix, filename: &str) -> std::io::Result<()> {
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

pub fn petsc_write_vector(vector: &Vector, filename: &str) -> std::io::Result<()> {
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

pub fn petsc_read_vector(filename: &str) -> std::io::Result<Vector> {
  let file = File::open(filename)?;
  let mut reader = BufReader::new(file);

  let magic = reader.read_i32::<BigEndian>()?;
  assert_eq!(magic, PETSC_VEC_FILE_CLASSID);

  let nrows = reader.read_i32::<BigEndian>()? as usize;

  let mut vector = Vector::zeros(nrows);
  for i in 0..nrows {
    vector[i] = reader.read_f64::<BigEndian>()?;
  }

  Ok(vector)
}

pub fn petsc_read_eigenvals(filename: &str) -> std::io::Result<Vector> {
  let file = File::open(filename)?;
  let mut reader = BufReader::new(file);

  let neigenvals = reader.read_i32::<BigEndian>()? as usize;
  let mut eigenvals = Vector::zeros(neigenvals);

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
  Ok(Matrix::from_column_slice(nrows, ncols, &data))
}

pub fn petsc_ghiep(lhs: &CsrMatrix, rhs: &CsrMatrix, neigen_values: usize) -> (Vector, Matrix) {
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

pub fn petsc_saddle_point(lhs: &CsrMatrix, rhs: &Vector) -> Vector {
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
