pub fn indicies_to_flags(indicies: &[usize], len: usize) -> Vec<bool> {
  let mut flags = vec![false; len];
  indicies.iter().for_each(|&i| flags[i] = true);
  flags
}

pub fn flags_to_indicies(flags: &[bool]) -> Vec<usize> {
  flags
    .iter()
    .enumerate()
    .filter_map(|(i, &flag)| flag.then_some(i))
    .collect()
}

pub fn sparse_to_dense_data<T>(sparse: Vec<(usize, T)>, len: usize) -> Vec<Option<T>> {
  let mut dense = Vec::from_iter((0..len).map(|_| None));
  sparse.into_iter().for_each(|(i, t)| dense[i] = Some(t));
  dense
}

pub fn dense_to_sparse_data<T>(dense: Vec<Option<T>>) -> Vec<(usize, T)> {
  dense
    .into_iter()
    .enumerate()
    .filter_map(|(i, o)| o.map(|v| (i, v)))
    .collect()
}

/// converts linear index to cartesian index
///
/// converts linear index in 0..dim_len^d to cartesian index in (0)^d..(dim_len)^d
pub fn linear_index2cartesian_index(
  mut lin_idx: usize,
  dim_len: usize,
  dim: usize,
) -> na::DVector<usize> {
  let mut cart_idx = na::DVector::zeros(dim);
  for icomp in 0..dim {
    cart_idx[icomp] = lin_idx % dim_len;
    lin_idx /= dim_len;
  }
  cart_idx
}

/// converts cartesian index to linear index
///
/// converts cartesian index in (0)^d..(dim_len)^d to linear index in 0..dim_len^d
pub fn cartesian_index2linear_index(cart_idx: na::DVector<usize>, dim_len: usize) -> usize {
  let dim = cart_idx.len();
  let mut lin_idx = 0;
  for icomp in (0..dim).rev() {
    lin_idx *= dim_len;
    lin_idx += cart_idx[icomp];
  }
  lin_idx
}

pub fn gram(m: &na::DMatrix<f64>) -> na::DMatrix<f64> {
  m.transpose() * m
}

pub fn gram_det(m: &na::DMatrix<f64>) -> f64 {
  gram(m).determinant()
}

pub fn gram_det_sqrt(m: &na::DMatrix<f64>) -> f64 {
  gram_det(m).sqrt()
}

pub fn faervec2navec(faer: &faer::Mat<f64>) -> na::DVector<f64> {
  assert!(faer.ncols() == 1);
  na::DVector::from_iterator(faer.nrows(), faer.row_iter().map(|r| r[0]))
}

pub fn navec2faervec(na: &na::DVector<f64>) -> faer::Mat<f64> {
  let mut faer = faer::Mat::zeros(na.nrows(), 1);
  for (i, &v) in na.iter().enumerate() {
    faer[(i, 0)] = v;
  }
  faer
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

// TODO: do it for sparse matrices directly, by computing the ratio between the
// largest and smallest eigenvalue.
pub fn condition_number(mat: na::DMatrix<f64>) -> f64 {
  mat.norm() * mat.try_inverse().unwrap().norm()
}
