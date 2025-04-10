use std::mem;

use crate::util::{CumsumExt, IterAllEqExt};

pub trait DMatrixExt {
  fn is_full_rank(&self, eps: f64) -> bool;
  fn is_spd(&self) -> bool;
  fn condition_number(self) -> f64;
}
impl DMatrixExt for na::DMatrix<f64> {
  fn is_full_rank(&self, eps: f64) -> bool {
    self.rank(eps) == self.nrows().min(self.ncols())
  }
  fn is_spd(&self) -> bool {
    self.is_square() && *self == self.transpose() && na::Cholesky::new(self.clone()).is_some()
  }
  fn condition_number(self) -> f64 {
    self.norm() * self.try_inverse().unwrap().norm()
  }
}

pub fn bilinear_form(mat: &nas::CsrMatrix<f64>, u: &na::DVector<f64>, v: &na::DVector<f64>) -> f64 {
  ((mat.transpose() * u).transpose() * v).x
}
pub fn quadratic_form_sparse(mat: &nas::CsrMatrix<f64>, u: &na::DVector<f64>) -> f64 {
  bilinear_form(mat, u, u)
}

pub fn kronecker_sum<T>(mats: &[na::DMatrix<T>]) -> na::DMatrix<T>
where
  T: na::Scalar + num_traits::Zero + num_traits::One + na::ClosedMulAssign + na::ClosedAddAssign,
{
  assert!(!mats.is_empty());
  assert!(mats.iter().all(|m| m.nrows() == m.ncols()));

  let eyes: Vec<_> = mats
    .iter()
    .map(|m| na::DMatrix::identity(m.nrows(), m.nrows()))
    .collect();

  let kron_size = mats.iter().map(|mat| mat.nrows()).product::<usize>();
  let mut kron_sum = na::DMatrix::zeros(kron_size, kron_size);
  for (dim, mat) in mats.iter().enumerate() {
    let eyes_before = eyes[..dim]
      .iter()
      .fold(na::DMatrix::identity(1, 1), |prod, eye| prod.kronecker(eye));
    let eyes_after = eyes[dim + 1..]
      .iter()
      .fold(na::DMatrix::identity(1, 1), |prod, eye| prod.kronecker(eye));

    let kron_prod = eyes_before.kronecker(mat).kronecker(&eyes_after);
    kron_sum += kron_prod;
  }

  kron_sum
}

pub fn matrix_from_const_diagonals<T>(
  values: &[T],
  offsets: &[isize],
  nrows: usize,
  ncols: usize,
) -> na::DMatrix<T>
where
  T: num_traits::Zero + na::Scalar + Copy,
{
  let mut matrix = na::DMatrix::zeros(nrows, ncols);

  for (idiag, &offset) in offsets.iter().enumerate() {
    let [start_row, start_col] = if offset >= 0 {
      [0, offset as usize]
    } else {
      [(-offset) as usize, 0]
    };

    let mut r = start_row;
    let mut c = start_col;
    while r < nrows && c < ncols {
      matrix[(r, c)] = values[idiag];
      r += 1;
      c += 1;
    }
  }

  matrix
}

pub fn save_vector(
  mu: &na::DVector<f64>,
  path: impl AsRef<std::path::Path>,
) -> std::io::Result<()> {
  use std::io::Write;
  let mut file = std::fs::File::create(path).unwrap();
  for v in mu.iter() {
    writeln!(file, "{v}")?;
  }
  Ok(())
}

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
      .unique_eq()
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
      .unique_eq()
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
