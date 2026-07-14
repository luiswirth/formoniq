pub mod coord;
pub mod metric;

use crate::Dim;

use common::{combo::factorial_f64, gramian::RiemannianMetric};

pub fn refsimp_vol(dim: Dim) -> f64 {
  factorial_f64(dim).recip()
}

/// The volume of a cell carrying the given metric tensor,
/// $vol(K) = vol(hat(K)) sqrt(det g)$.
pub fn cell_volume(metric: &RiemannianMetric) -> f64 {
  refsimp_vol(metric.dim()) * metric.det_sqrt()
}
