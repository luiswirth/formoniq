pub mod coord;
pub mod metric;

use crate::Dim;

use common::combo::factorialf;

pub fn refsimp_vol(dim: Dim) -> f64 {
  factorialf(dim).recip()
}
