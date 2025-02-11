use super::{
  handle::{CellHandle, KSimplexIdx},
  REFERENCE_COMPLEXES,
};
use crate::Dim;

pub type LocalSkeleton = Vec<KSimplexIdx>;

#[derive(Debug, Clone)]
pub struct LocalComplex {
  skeletons: Vec<LocalSkeleton>,
}
impl LocalComplex {
  pub fn new(skeletons: Vec<LocalSkeleton>) -> Self {
    Self { skeletons }
  }

  pub fn from_cell(cell: CellHandle) -> Self {
    let subs = (0..=cell.dim())
      .map(|dim_sub| cell.subsimps(dim_sub).map(|sub| sub.kidx()).collect())
      .collect();
    Self::new(subs)
  }

  pub fn dim(&self) -> Dim {
    self.skeletons.len() - 1
  }
  pub fn nvertices(&self) -> usize {
    self.vertices().len()
  }
  pub fn vertices(&self) -> &[KSimplexIdx] {
    &self.skeletons[0]
  }
  pub fn skeletons(&self) -> &[Vec<KSimplexIdx>] {
    &self.skeletons
  }

  pub fn boundary_operator(&self, dim_sub: Dim) -> na::DMatrix<f64> {
    // TODO: precompute and store
    REFERENCE_COMPLEXES[self.dim()]
      .boundary_operator(dim_sub)
      .to_nalgebra_dense()
  }
}
