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
