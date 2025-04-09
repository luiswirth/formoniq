pub struct AffineTransform {
  pub translation: na::DVector<f64>,
  pub linear: na::DMatrix<f64>,
}
impl AffineTransform {
  pub fn new(translation: na::DVector<f64>, linear: na::DMatrix<f64>) -> Self {
    Self {
      translation,
      linear,
    }
  }

  pub fn apply_forward(&self, coord: na::DVectorView<f64>) -> na::DVector<f64> {
    &self.linear * coord + &self.translation
  }
  pub fn apply_backward(&self, coord: na::DVectorView<f64>) -> na::DVector<f64> {
    if self.linear.is_empty() {
      return na::DVector::default();
    }
    self
      .linear
      .clone()
      .svd(true, true)
      .solve(&(coord - &self.translation), 1e-12)
      .unwrap()
  }

  pub fn pseudo_inverse(&self) -> Self {
    let linear = self.linear.clone().pseudo_inverse(1e-12).unwrap();
    let translation = &linear * &self.translation;
    Self {
      translation,
      linear,
    }
  }
}
