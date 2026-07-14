//! The geometry a mesh carries: the metric layer, and the extrinsic coordinate
//! layer downstream of it.
//!
//! [`metric`] is the intrinsic one and the only one FEEC assembly consumes;
//! [`coord`] is an embedding, one [`Geometry`](metric::Geometry) implementor
//! among several. The dependency runs that way and not the other: an embedding
//! induces a metric, a metric induces no embedding.

pub mod coord;
pub mod metric;

use crate::atlas::refsimp_vol;

use common::gramian::RiemannianMetric;

/// The volume of a cell carrying the given metric tensor,
/// $vol(K) = vol(hat(K)) sqrt(det g)$.
///
/// The chart contributes [`refsimp_vol`], the metric the factor
/// $sqrt(det g)$ -- the whole of the geometry, in one scalar.
pub fn cell_volume(metric: &RiemannianMetric) -> f64 {
  refsimp_vol(metric.dim()) * metric.det_sqrt()
}
