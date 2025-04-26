use crate::util::{CumsumExt, IterAllEqExt};

use std::mem;

pub type Vector<T = f64> = na::DVector<T>;
pub type RowVector<T = f64> = na::RowDVector<T>;
pub type Matrix<T = f64> = na::DMatrix<T>;

pub type VectorView<'a, T = f64> = na::DVectorView<'a, T>;
pub type RowVectorView<'a, T = f64> = na::MatrixView<'a, T, na::U1, na::Dyn>;
pub type MatrixView<'a, T = f64> = na::DMatrixView<'a, T>;

pub type CooMatrix<T = f64> = nas::CooMatrix<T>;
pub type CsrMatrix<T = f64> = nas::CsrMatrix<T>;
pub type CscMatrix<T = f64> = nas::CscMatrix<T>;

pub trait MatrixExt {
  fn is_full_rank(&self, eps: f64) -> bool;
  fn is_spd(&self) -> bool;
  fn condition_number(self) -> f64;
}
impl MatrixExt for Matrix {
  fn is_full_rank(&self, eps: f64) -> bool {
    if self.is_empty() {
      return true;
    }
    self.rank(eps) == self.nrows().min(self.ncols())
  }
  fn is_spd(&self) -> bool {
    na::Cholesky::new(self.clone()).is_some()
  }
  fn condition_number(self) -> f64 {
    self.norm() * self.try_inverse().unwrap().norm()
  }
}

pub fn bilinear_form_sparse(mat: &CsrMatrix, u: &Vector, v: &Vector) -> f64 {
  //u.transpose() * mass * u
  ((mat.transpose() * u).transpose() * v).x
}
pub fn quadratic_form_sparse(mat: &CsrMatrix, u: &Vector) -> f64 {
  bilinear_form_sparse(mat, u, u)
}

pub fn kronecker_sum<T>(mats: &[Matrix<T>]) -> Matrix<T>
where
  T: na::Scalar + num_traits::Zero + num_traits::One + na::ClosedMulAssign + na::ClosedAddAssign,
{
  assert!(!mats.is_empty());
  assert!(mats.iter().all(|m| m.nrows() == m.ncols()));

  let eyes: Vec<_> = mats
    .iter()
    .map(|m| Matrix::identity(m.nrows(), m.nrows()))
    .collect();

  let kron_size = mats.iter().map(|mat| mat.nrows()).product::<usize>();
  let mut kron_sum = Matrix::zeros(kron_size, kron_size);
  for (dim, mat) in mats.iter().enumerate() {
    let eyes_before = eyes[..dim]
      .iter()
      .fold(Matrix::identity(1, 1), |prod, eye| prod.kronecker(eye));
    let eyes_after = eyes[dim + 1..]
      .iter()
      .fold(Matrix::identity(1, 1), |prod, eye| prod.kronecker(eye));

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
) -> Matrix<T>
where
  T: num_traits::Zero + na::Scalar + Copy,
{
  let mut matrix = Matrix::zeros(nrows, ncols);

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

pub fn save_vector(mu: &Vector, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
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

impl CooMatrixExt for CooMatrix {
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
