use std::{
  fs::File,
  io::{BufReader, BufWriter, Write},
};

use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};

#[derive(Default, Debug, Clone)]
pub struct SparseMatrix {
  nrows: usize,
  ncols: usize,
  triplets: Vec<(usize, usize, f64)>,
}

impl SparseMatrix {
  pub fn zeros(nrows: usize, ncols: usize) -> Self {
    Self::new(nrows, ncols, Vec::new())
  }
  pub fn new(nrows: usize, ncols: usize, triplets: Vec<(usize, usize, f64)>) -> Self {
    Self {
      nrows,
      ncols,
      triplets,
    }
  }

  pub fn nrows(&self) -> usize {
    self.nrows
  }
  pub fn ncols(&self) -> usize {
    self.ncols
  }
  pub fn triplets(&self) -> &[(usize, usize, f64)] {
    &self.triplets
  }

  pub fn into_parts(self) -> (usize, usize, Vec<(usize, usize, f64)>) {
    (self.nrows, self.ncols, self.triplets)
  }

  pub fn push(&mut self, r: usize, c: usize, v: f64) {
    assert!(r <= self.nrows() && c <= self.ncols());
    if v != 0.0 {
      self.triplets.push((r, c, v));
    }
  }

  pub fn set_zero<F>(&mut self, predicate: F)
  where
    F: Fn(usize, usize) -> bool,
  {
    let mut i = 0;
    while i < self.triplets.len() {
      let triplet = self.triplets[i];
      let r = triplet.0;
      let c = triplet.1;
      if predicate(r, c) {
        self.triplets.swap_remove(i);
      } else {
        i += 1;
      }
    }
  }

  pub fn to_nalgebra_coo(&self) -> nas::CooMatrix<f64> {
    let rows = self.triplets.iter().map(|t| t.0).collect();
    let cols = self.triplets.iter().map(|t| t.1).collect();
    let vals = self.triplets.iter().map(|t| t.2).collect();
    nas::CooMatrix::try_from_triplets(self.nrows, self.ncols, rows, cols, vals).unwrap()
  }

  pub fn to_nalgebra_csr(&self) -> nas::CsrMatrix<f64> {
    (&self.to_nalgebra_coo()).into()
  }

  pub fn to_nalgebra_dense(&self) -> na::DMatrix<f64> {
    (&self.to_nalgebra_coo()).into()
  }

  pub fn to_faer_csc(&self) -> faer::sparse::SparseColMat<usize, f64> {
    faer::sparse::SparseColMat::try_new_from_triplets(self.nrows, self.ncols, &self.triplets)
      .unwrap()
  }

  /// Returns `None` if matrix is not diagonal.
  pub fn try_into_diagonal(self) -> Option<na::DVector<f64>> {
    let mut diagonal = na::DVector::zeros(self.nrows.max(self.ncols));
    for (r, c, v) in self.triplets {
      if r == c {
        diagonal[r] += v;
      } else {
        println!("not diag ({r},{c})");
        return None;
      }
    }
    Some(diagonal)
  }

  pub fn mul_left_by_diagonal(&self, diagonal: &na::DVector<f64>) -> Self {
    let triplets = self
      .triplets
      .iter()
      .map(|&(r, c, mut v)| {
        v *= diagonal[r];
        (r, c, v)
      })
      .collect();
    Self::new(self.nrows, self.ncols, triplets)
  }

  pub fn transpose(&self) -> SparseMatrix {
    let mut triplets = self.triplets.clone();
    for t in &mut triplets {
      std::mem::swap(&mut t.0, &mut t.1);
    }
    Self::new(self.ncols, self.nrows, triplets)
  }
}

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
  let solver_path = "/home/luis/thesis/solvers/petsc-solver";
  petsc_write_matrix(lhs, &format!("{solver_path}/in/A.bin")).unwrap();
  petsc_write_matrix(rhs, &format!("{solver_path}/in/B.bin")).unwrap();

  let binary = "./ghiep";
  #[rustfmt::skip]
  let args = [
    "-st_pc_factor_mat_solver_type", "mumps",
    "-st_type", "sinvert",
    "-st_shift", "0.1",
    "-eps_target", "0.",
    //"-eps_gen_non_hermitian",
    "-eps_nev", &neigen_values.to_string(),
  ];

  let status = std::process::Command::new(binary)
    .current_dir(solver_path)
    .args(args)
    .status()
    .unwrap();
  assert!(status.success());

  let eigenvals = petsc_read_eigenvals(&format!("{solver_path}/out/eigenvals.bin")).unwrap();
  let eigenvecs = petsc_read_eigenvecs(&format!("{solver_path}/out/eigenvecs.bin")).unwrap();

  (eigenvals, eigenvecs)
}

pub fn petsc_saddle_point(lhs: &nas::CsrMatrix<f64>, rhs: &na::DVector<f64>) -> na::DVector<f64> {
  let solver_path = "/home/luis/thesis/solvers/petsc-solver";

  petsc_write_matrix(lhs, &format!("{solver_path}/in/A.bin")).unwrap();
  petsc_write_vector(rhs, &format!("{solver_path}/in/b.bin")).unwrap();

  let binary = "./hils";
  #[rustfmt::skip]
  let args: [&str; 0] = [
    //"-ksp_type", "minres",
    //"-pc_type", "lu",
    //"-ksp_max_it", "1000",
    //"-ksp_rtol", "1e-9",
  ];

  let status = std::process::Command::new(binary)
    .current_dir(solver_path)
    .args(args)
    .status()
    .unwrap();
  assert!(status.success());

  petsc_read_vector(&format!("{solver_path}/out/x.bin")).unwrap()
}
