//! Dense and sparse nalgebra type aliases, and the block-matrix builder that
//! assembly needs on top of them.
//!
//! Not a shared base crate on purpose: `simplicial` is the lowest crate that
//! genuinely needs sparse matrices (the boundary operators), so this is where
//! the aliases live; `derham` and `formoniq` reuse them because they already
//! depend on `simplicial` for real reasons, not because a `Vector`/`Matrix`
//! alias is worth a crate of its own.

use std::mem;

trait CumsumExt {
  fn cumsum(self) -> impl Iterator<Item = usize>;
}
impl<I: IntoIterator<Item = usize>> CumsumExt for I {
  fn cumsum(self) -> impl Iterator<Item = usize> {
    self.into_iter().scan(0, |acc, x| {
      *acc += x;
      Some(*acc)
    })
  }
}

trait IterAllEqExt<T> {
  fn unique_eq(self) -> Option<T>;
}
impl<T: PartialEq, I: IntoIterator<Item = T>> IterAllEqExt<T> for I {
  fn unique_eq(self) -> Option<T> {
    let mut iter = self.into_iter();
    let first = iter.next()?;
    iter.all(|elem| elem == first).then_some(first)
  }
}

pub type Vector<T = f64> = na::DVector<T>;
pub type RowVector<T = f64> = na::RowDVector<T>;
pub type Matrix<T = f64> = na::DMatrix<T>;

pub type VectorView<'a, T = f64> = na::DVectorView<'a, T>;
pub type RowVectorView<'a, T = f64> = na::MatrixView<'a, T, na::U1, na::Dyn>;

pub type CooMatrix<T = f64> = nas::CooMatrix<T>;
pub type CsrMatrix<T = f64> = nas::CsrMatrix<T>;

pub trait CooMatrixExt {
  fn neg(self) -> Self;
  fn block(block_grid: &[&[&Self]]) -> Self;
  fn grow(&mut self, nrows_added: usize, ncols_added: usize);
  fn transpose(self) -> Self;
}

impl CooMatrixExt for CooMatrix {
  fn grow(&mut self, nrows_added: usize, ncols_added: usize) {
    let nrows = self.nrows() + nrows_added;
    let ncols = self.ncols() + ncols_added;
    let (rows, cols, values) = mem::replace(self, Self::new(0, 0)).disassemble();
    *self = Self::try_from_triplets(nrows, ncols, rows, cols, values).unwrap();
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
