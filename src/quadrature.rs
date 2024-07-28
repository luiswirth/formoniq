use crate::{
  geometry::{ref_vol, GeometrySimplex},
  Dim,
};

/// A quadrature rule defined on the reference simplex
pub struct QuadRule {
  nodes: na::DMatrix<f64>,
  weights: na::DVector<f64>,
}
impl QuadRule {
  pub fn dim(&self) -> Dim {
    self.nodes.ncols()
  }
  pub fn ref_simp(&self) -> GeometrySimplex {
    GeometrySimplex::new_ref(self.dim())
  }
  pub fn apply_ref<F>(&self, f: F) -> f64
  where
    F: Fn(na::DVectorView<f64>) -> f64,
  {
    self
      .nodes
      .column_iter()
      .zip(self.weights.iter())
      .map(|(n, w)| w * f(n))
      .sum()
  }
  pub fn apply<F>(&self, f: F, simp: GeometrySimplex) -> f64
  where
    F: Fn(na::DVectorView<f64>) -> f64,
  {
    let nodes_trans = simp.reference_transform().apply(self.nodes.as_view());
    simp.vol() / ref_vol(simp.dim_intrinsic())
      * nodes_trans
        .column_iter()
        .zip(self.weights.iter())
        .map(|(n, w)| w * f(n))
        .sum::<f64>()
  }
}
