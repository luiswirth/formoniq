extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

fn main() {
  let grid_sizes = vec![2, 2, 2];
  let laplacian = laplacian_nd(&grid_sizes);
  println!("{laplacian}");
}

fn laplacian_nd(sizes: &[usize]) -> na::DMatrix<f64> {
  let lapls: Vec<_> = sizes.iter().map(|&size| laplacian_1d(size)).collect();
  kronecker_sum(&lapls)
}

fn laplacian_1d(size: usize) -> na::DMatrix<f64> {
  let stencil = [-1.0, 2.0, -1.0];
  matrix_from_diagonals(&stencil[..], &[-1, 0, 1], size, size)
}

fn kronecker_sum(mats: &[na::DMatrix<f64>]) -> na::DMatrix<f64> {
  assert!(!mats.is_empty());
  assert!(mats.iter().all(|m| m.nrows() == m.ncols()));

  let kron_size = mats.iter().map(|mat| mat.nrows()).product::<usize>();
  let mut kron_sum = na::DMatrix::zeros(kron_size, kron_size);
  for (dim, mat) in mats.iter().enumerate() {
    let eyes_before = mats[..dim]
      .iter()
      .map(|m| na::DMatrix::identity(m.nrows(), m.nrows()))
      .fold(na::DMatrix::identity(1, 1), |acc, x| acc.kronecker(&x));
    let eyes_after = mats[dim + 1..]
      .iter()
      .map(|m| na::DMatrix::identity(m.nrows(), m.nrows()))
      .fold(na::DMatrix::identity(1, 1), |acc, x| acc.kronecker(&x));

    let kron_prod = eyes_before.kronecker(mat).kronecker(&eyes_after);
    kron_sum += kron_prod;
  }

  kron_sum
}

fn matrix_from_diagonals(
  values: &[f64],
  offsets: &[isize],
  nrows: usize,
  ncols: usize,
) -> na::DMatrix<f64> {
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
