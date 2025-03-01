pub trait DMatrixExt {
  fn gramian(&self) -> Self;
  fn gram_det(&self) -> f64;
  fn gram_det_sqrt(&self) -> f64;
  fn condition_number(self) -> f64;
}
impl DMatrixExt for na::DMatrix<f64> {
  fn gramian(&self) -> Self {
    self.transpose() * self
  }
  fn gram_det(&self) -> f64 {
    self.gramian().determinant()
  }
  fn gram_det_sqrt(&self) -> f64 {
    self.gram_det().sqrt()
  }

  // TODO: also support sparse matrices
  // but by computing the ratio between the largest and smallest eigenvalue.
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

pub fn are_f64_eq(a: f64, b: f64, eps: Option<f64>) -> bool {
  let eps = eps.unwrap_or(1e-12);
  let diff = a - b;
  diff <= eps
}

pub fn assert_f64_eq(a: f64, b: f64, eps: Option<f64>) {
  if !are_f64_eq(a, b, eps) {
    let diff = a - b;
    println!("a={a:.3}");
    println!("b={b:.3}");
    println!("a-b={diff:.3}");
    panic!("Matrices not equal.");
  }
}

pub fn are_mat_eq(a: &na::DMatrix<f64>, b: &na::DMatrix<f64>, eps: Option<f64>) -> bool {
  let eps = eps.unwrap_or(1e-12);
  let diff = a - b;
  let error = diff.norm();
  error <= eps
}

pub fn assert_mat_eq(a: &na::DMatrix<f64>, b: &na::DMatrix<f64>, eps: Option<f64>) {
  if !are_mat_eq(a, b, eps) {
    let diff = a - b;
    println!("a={a:.3}");
    println!("b={b:.3}");
    println!("a-b={diff:.3}");
    panic!("Matrices not equal.");
  }
}
