pub mod coord;
pub mod metric;

use crate::Dim;

use common::combo::factorial_f64;

pub fn refsimp_vol(dim: Dim) -> f64 {
  factorial_f64(dim).recip()
}
