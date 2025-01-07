use super::{ref_vol, MetricComplex};
use crate::metric::{EdgeLengths, RiemannianMetric};

use topology::{complex::local::LocalComplex, Dim};

#[derive(Debug, Clone)]
pub struct LocalMetricComplex {
  topology: LocalComplex,
  edge_lengths: EdgeLengths,
  metric: RiemannianMetric,
}
impl LocalMetricComplex {
  pub fn new(topology: LocalComplex, edge_lengths: EdgeLengths) -> Self {
    let metric = edge_lengths.to_regge_metric(topology.dim());
    Self {
      topology,
      edge_lengths,
      metric,
    }
  }

  pub fn reference(dim: Dim) -> Self {
    let global = MetricComplex::reference(dim);
    let facet = global.topology.facets().get_by_kidx(0);
    global.local_complex(facet)
  }

  pub fn topology(&self) -> &LocalComplex {
    &self.topology
  }
  pub fn edge_lengths(&self) -> &EdgeLengths {
    &self.edge_lengths
  }
  pub fn metric(&self) -> &RiemannianMetric {
    &self.metric
  }

  pub fn dim(&self) -> Dim {
    self.topology.dim()
  }

  /// The volume of this cell.
  pub fn vol(&self) -> f64 {
    ref_vol(self.dim()) * self.metric().det_sqrt()
  }

  /// The diameter of this cell.
  /// This is the maximum distance of two points inside the cell.
  pub fn diameter(&self) -> f64 {
    self
      .edge_lengths
      .iter()
      .copied()
      .max_by(|a, b| a.partial_cmp(b).unwrap())
      .unwrap()
  }

  /// The shape regularity measure of this cell.
  pub fn shape_reguarity_measure(&self) -> f64 {
    self.diameter().powi(self.dim() as i32) / self.vol()
  }
}
