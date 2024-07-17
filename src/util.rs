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

pub fn factorial(num: usize) -> usize {
  (1..=num).product()
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

pub fn sort_count_swaps<T: Ord>(a: &mut [T]) -> usize {
  let mut nswaps = 0;

  let mut n = a.len();
  let mut swapped = true;
  while swapped {
    swapped = false;
    for i in 1..n {
      if a[i - 1] > a[i] {
        a.swap(i - 1, i);
        swapped = true;
        nswaps += 1;
      }
    }
    n -= 1;
  }
  nswaps
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
