use crate::linalg::nalgebra::{Matrix, Vector, VectorView};

pub struct AffineTransform {
  pub translation: Vector,
  pub linear: Matrix,
}
impl AffineTransform {
  pub fn new(translation: Vector, linear: Matrix) -> Self {
    Self {
      translation,
      linear,
    }
  }

  pub fn dim_domain(&self) -> usize {
    self.linear.ncols()
  }
  pub fn dim_image(&self) -> usize {
    self.linear.nrows()
  }

  pub fn apply_forward(&self, coord: VectorView) -> Vector {
    &self.linear * coord + &self.translation
  }
  pub fn apply_backward(&self, coord: VectorView) -> Vector {
    if self.linear.is_empty() {
      return Vector::default();
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
