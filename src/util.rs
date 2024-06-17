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
